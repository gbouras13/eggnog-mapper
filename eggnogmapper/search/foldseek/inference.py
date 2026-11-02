from Bio import SeqIO
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
import numpy as np
from typing import Any, Dict, List,  Tuple, Union
import csv
import sys
import math
import hashlib

from ...emapperException import EmapperException
from .profiles import generate_mmseqs_db_from_aa, computeLogPSSM, toBuffer_pssm, \
    toBuffer_pssm_states, write_ss12_profile_db, BACKPROBS_12ST, PROFILE_RECORD_END, \
    build_database_seq, copy_and_create_extras, sort_index_file, remove_file
from .create_fs_db import ALPHABET_3DI, ALPHABET_12ST, clean_aa_sequence

# ModernProst checkpoint used to predict the structural alphabets from sequence.
# ModernProst3Di12StModel has two heads and returns logits as a dict:
#   {"3di": (B, L, 20), "12st": (B, L, 12)}
# so a single forward pass yields both alphabets Foldseek needs.
MODERNPROST_MODEL = "gbouras13/modernprost-50M-pssm"

# 3Di index -> letter. Only the 20 real states are ever written: eggnog-mapper
# does not mask low-confidence positions (see write_predictions).
SS_MAPPING = {i: c for i, c in enumerate(ALPHABET_3DI)}
# 12-state index -> letter. This alphabet has no 'X'.
ST12_MAPPING = {i: c for i, c in enumerate(ALPHABET_12ST)}


def write_predictions(
    predictions: Dict[str, Dict[str, Tuple[List[str], Any, Any]]],
    out_path: Path,
    out_path_12st: Path = None
) -> None:
    """
    Write predictions to an output file.

    Low-confidence 3Di positions are deliberately NOT masked to 'X'. Masking
    is not meaningful for the combined 3Di + 12-state encoding: the 12-state
    head has no masked state of its own, so a masked 3Di position would still
    carry a confident 12-state call, and the two alphabets would disagree about
    the same residue. It also keeps the written 3Di alphabet to the 20 real
    states, which is what the packed encoding in create_fs_db assumes.

    Args:
        predictions: per-sequence tuple of
            (pred_3di, mean_prob, max_prob, all_prob, pred_12st, all_prob_12st).
        out_path (Path): Path to the 3Di output FASTA.
        out_path_12st (Path): Path to the 12-state output FASTA. Skipped when the
            model did not produce a 12-state head.

    Returns:
        None
    """
    prediction_dict = {
            k: v for k, v in predictions.items() if len(v) > 0
        }

    has_12st = any(v[4] is not None for v in prediction_dict.values())

    out_f12 = open(out_path_12st, "w") if (out_path_12st and has_12st) else None
    try:
        with open(out_path, "w") as out_f:
            for seq_id, (pred, mean_prob, max_prob, all_prob, pred12, all_prob12) in prediction_dict.items():
                seq = "".join(SS_MAPPING[int(yhat)] for yhat in pred)

                out_f.write(f">{seq_id}\n{seq}\n")

                if out_f12 is not None and pred12 is not None:
                    seq12 = "".join(ST12_MAPPING[int(yhat)] for yhat in pred12)
                    out_f12.write(f">{seq_id}\n{seq12}\n")
    finally:
        if out_f12 is not None:
            out_f12.close()

    print(f"Finished writing results to {out_path}")
    if out_f12 is not None:
        print(f"Finished writing 12-state results to {out_path_12st}")
    return None

def split_logits(logits) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits_3di, logits_12st) from a model's logits output.

    ModernProst3Di12StModel returns a dict of both heads. Single-head
    checkpoints return a plain tensor of 3Di logits, in which case there is no
    12-state channel and the second element is None.
    """
    if isinstance(logits, dict):
        try:
            return logits["3di"], logits.get("12st")
        except KeyError:
            raise EmapperException(
                f"ModernProst model returned logits keys {sorted(logits)}; "
                f"expected a '3di' key."
            )
    return logits, None


def chunk_sequence(seq, max_len):
    """
    Yield (start, subseq) splitting seq into `max_len` nearly equal chunks.
    No overlap. Preserves order.
    """
    L = len(seq)
    n_chunks = math.ceil(L / max_len)
    base = L // n_chunks
    remainder = L % n_chunks

    start = 0
    for i in range(n_chunks):
        # distribute the extra 1 from remainder to the first `remainder` chunks
        size = base + (1 if i < remainder else 0)
        end = start + size
        yield start, seq[start:end]
        start = end


def toCPU(tensor: torch.Tensor) -> np.ndarray:
    """
    Move a tensor to CPU and convert it to a NumPy array.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: NumPy array.
    """
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    else:
        return tensor.detach().cpu().numpy()


def calc_md5_sum(tarball_path: Path, buffer_size: int = 1024 * 1024) -> str:
    """
    Calculate the MD5 checksum of the given file.

    Args:
        tarball_path (Path): The path to the file for which the MD5 checksum needs to be calculated.
        buffer_size (int): The buffer size for reading the file.

    Returns:
        str: The MD5 checksum of the file.
    """

    md5 = hashlib.md5()
    with tarball_path.open("rb") as fh:
        data = fh.read(buffer_size)
        while data:
            md5.update(data)
            data = fh.read(buffer_size)
    return md5.hexdigest()

def check_model_download(model_dir: Path, model_name: str = MODERNPROST_MODEL) -> bool:
    """Decide whether the model still has to be fetched from Hugging Face.

    Args:
        model_dir (Path): Hugging Face cache directory.
        model_name (str): Repo id of the model.
    Returns:
        bool: True when the model is not in the cache yet.

    We only check that a complete snapshot is present; Hugging Face verifies
    file integrity against the remote ETags itself when it populates the cache.
    """
    model_sub_dir = "models--" + model_name.replace("/", "--")
    snapshots = Path(model_dir) / model_sub_dir / "snapshots"

    if not snapshots.is_dir():
        print(f"{model_name} not found in {model_dir}.")
        return True

    # a snapshot dir exists; require the weights and the config to be resolvable
    for snapshot in snapshots.iterdir():
        has_config = (snapshot / "config.json").exists()
        has_weights = any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.bin"))
        if has_config and has_weights:
            return False

    print(f"{model_name} cache in {model_dir} is incomplete.")
    return True

def infer_3Di(in_fasta, output_dir, cache_dir, batch_size, max_batch_residues,
              profiles, model_name=MODERNPROST_MODEL):

    # Dictionary to store the records
    cds_dict = {}

    # might need a better indexing style solution for massive FASTAs
    with open(in_fasta, "rt") as handle:  
        for record in SeqIO.parse(handle, "fasta"):
            cds_dict[record.id] = str(record.seq)

    if not cds_dict:
        print(f"No amino acid sequences found in {in_fasta}") # dont use logger, use print to match the rest of it

    # Drop trailing stop codons and normalise internal ones; create_fs_db
    # applies the same function to the amino-acid database. Done before the
    # --profiles amino-acid database is built from cds_dict, so the AA, 3Di and
    # 12-state profiles of a query all have one column per residue.
    # Residues the tokenizer has no token for would otherwise become <unk>,
    # which the model was not trained on.
    cds_dict = {k: clean_aa_sequence(v).replace("U", "X").replace("Z", "X").replace("O", "X")
                for k, v in cds_dict.items()}

    sys.path.append(cache_dir)


    # One checkpoint serves both paths: its 3Di head gives the discrete
    # sequence and the per-position distribution used for the PSSM profiles,
    # and its 12st head gives the second structural alphabet.
    #
    # The same checkpoint must be used for the target database and for the
    # queries searched against it — the two alphabets are model-specific, so
    # mixing checkpoints silently degrades the alignment scores.

    if profiles:
        prefix = "query"

        lookup = generate_mmseqs_db_from_aa(cds_dict, output_dir, prefix)

        mmseqs2_db_dir: Path = Path(output_dir) / f"query_profiledb" # this will be subdir where the dummy mmseqs2 db is 
        mmseqs2_db_dir.mkdir(parents=True, exist_ok=True)

        # create MMSeqs2 db names
        short_db_name = f"{prefix}"
        mmseqs_db: Path = Path(mmseqs2_db_dir) / f"{short_db_name}"
        mmseqs_db_lookup: Path = Path(mmseqs2_db_dir) / f"{short_db_name}.lookup"


        seq_output_db: Path = Path(mmseqs2_db_dir) / f"{short_db_name}_profile"
        seq_index = f"{seq_output_db}.index"
        pssm_output_db: Path = Path(mmseqs2_db_dir) / f"{short_db_name}_profile_ss"
        pssm_index = f"{pssm_output_db}.index"
        
        # lookup = build_lookup(mmseqs_db_lookup)

        pssm_db_builder = {
            "dummy_seq_db": mmseqs_db,
            "lookup": lookup,
            "parts": [],
            "index_lines": [],
            "offset": 0,
            "output_db": pssm_output_db,
            "index_file": pssm_index,
            "seq_output_db": seq_output_db,
            "seq_index_file": seq_index
        }

    localfile = True
    download = False

    download = check_model_download(cache_dir, model_name)

    if download is True:
        localfile = False
        print(f"{model_name} not found. Downloading {model_name} from Hugging Face")

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
        force_download=download,
        local_files_only=localfile
    )

    # half precision
    model.half()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
        force_download=download,
        local_files_only=localfile
    )



    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} Loaded")
    print(f"{model_name} Total Parameters: {total_params}")

    predictions = {}
    batch_predictions = {}
    chunk_store = {}

    fail_ids = []

    valid_cds_dict = {}

    for k, v in cds_dict.items():
        if v and len(v) > 0 and isinstance(v[0], str):
            valid_cds_dict[k] = v
        else:
            print(f"Protein with header {k} is corrupt. It will be saved in fails.tsv")
            fail_ids.append(k)

    # Sort to catch OOM - only the valid ones
    # sort the original keys by length - longest first
    original_keys = list(cds_dict.keys())


    cds_dict = dict(
        sorted(valid_cds_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    )

    batch = list()
    # batch_size = 5
    # max_batch_residues = 50000
    max_seq_len = 50000


    batch = list()
    n_res_batch = 0
    chunk_len = 1568 # set to 1568 chunk length

    for seq_idx, (id, seq) in tqdm(enumerate(cds_dict.items(), 1), total=len(cds_dict), desc="Predicting 3Di"):
        # already cleaned above
        slen = len(seq)


        if slen > chunk_len:
                    for chunk_idx, (start, subseq) in enumerate(chunk_sequence(seq, chunk_len)): # hardcoded to 1568
                        chunk_pid = f"{id}__chunk{chunk_idx}"
                        batch.append((chunk_pid, subseq, len(subseq)))
                        n_res_batch += len(subseq)
        else:
            batch.append((id, seq, slen))
            n_res_batch += slen

        if (
            len(batch) >= batch_size
            or n_res_batch >= max_batch_residues
            or seq_idx == len(cds_dict)
            or slen > max_seq_len
        ):
            ids, seqs, seq_lens = zip(*batch)
            batch = list()
            batch.clear()
            n_res_batch = 0
            batch = list()

            # fix for v5 - adding special tokens is wrong as it appends None to start and end of each seq (didnt in v4)
            # this works for both

            # inputs = tokenizer.batch_encode_plus(
            #     seqs,
            #     add_special_tokens=False,
            #     padding="longest",
            #     return_tensors="pt",
            # ).to(device)

            inputs = tokenizer(
                seqs,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt"
                ).to(device)

            
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except RuntimeError:
                print(f" number of residues in batch {n_res_batch}")
                print(f" seq length is {slen}")
                print(f" ids are {ids}")
                print(
                    "RuntimeError during embedding for {} (L={})".format(
                        id, slen
                    )
                )
                for id in ids:
                    fail_ids.append(id)
                continue

    
            try:
                logits_3di, logits_12st = split_logits(outputs.logits)

                softmax_logits = F.softmax(logits_3di, dim=-1)
                all_probabilities = toCPU(softmax_logits)  # [B, L, C] - won't emit here
                max_probabilities = toCPU(
                    softmax_logits.max(dim=-1, keepdim=True).values
                )  # [B, L, 1]


                # ---- predictions (argmax over classes) ----
                # argmax
                preds = torch.argmax(logits_3di, dim=-1, keepdim=True)  # [B, L, 1]
                preds = toCPU(preds).astype(np.byte)

                if logits_12st is not None:
                    preds12 = torch.argmax(logits_12st, dim=-1, keepdim=True)  # [B, L, 1]
                    preds12 = toCPU(preds12).astype(np.byte)
                    # the full 12-state distribution is only needed to build the
                    # _ss12 profile database
                    all_probabilities_12st = (
                        toCPU(F.softmax(logits_12st, dim=-1)) if profiles else None)
                else:
                    preds12 = None
                    all_probabilities_12st = None

                for batch_idx, identifier in enumerate(ids):

                    s_len = seq_lens[batch_idx]

                    # predictions
                    pred = preds[batch_idx, :s_len]          # (L, 1)
                    max_prob = max_probabilities[batch_idx, :s_len]  # (L, 1)
                    # The full (L, 20) distribution is only needed to build the
                    # PSSM profiles. Holding it for every query costs ~20 bytes
                    # per residue (6 GB for a 400k-protein input), so keep it
                    # only when --profiles will actually consume it.
                    all_prob = all_probabilities[batch_idx, :s_len] if profiles else None
                    pred12 = preds12[batch_idx, :s_len] if preds12 is not None else None
                    all_prob12 = (all_probabilities_12st[batch_idx, :s_len]
                                  if all_probabilities_12st is not None else None)


                    if "__chunk" in identifier:
                        base_id, chunk_tag = identifier.split("__chunk")
                        chunk_idx = int(chunk_tag)

                        if base_id not in chunk_store:
                            chunk_store[base_id] = {}

                        chunk_store[base_id][chunk_idx] = (
                                pred,
                                max_prob,
                                all_prob,
                                pred12,
                                all_prob12)

                        try:
                            len(chunk_store[base_id][chunk_idx][0])
                        except:
                            print(
                                f"{identifier} prediction has length 0"
                            )
                            fail_ids.append(base_id)
                            continue

                        if s_len != len(chunk_store[base_id][chunk_idx][0]):
                            print(
                                f"Length mismatch for {identifier}: is:{len(chunk_store[base_id][chunk_idx][0])} vs should:{s_len}"
                            )
                        
                    else:
                        mean_prob = round(100 * max_probabilities[batch_idx, 0:s_len].mean().item(), 2)
                    

                        batch_predictions[identifier] = (
                            pred,
                            mean_prob,
                            max_prob,
                            all_prob,
                            pred12,
                            all_prob12
                        )

                        try:
                            len(batch_predictions[identifier][0])
                        except:
                            print(
                                f"{identifier}  prediction has length 0"
                            )
                            fail_ids.append(identifier)
                            continue

                        if s_len != len(batch_predictions[identifier][0]):
                            print(
                                f"Length mismatch for {identifier}: is:{len(batch_predictions[identifier][0])} vs should:{s_len}"
                            )



            except IndexError:
                print(
                    "Index error during prediction for {} (L={})".format(
                        id, slen
                    )
                )

                for id in ids:
                    fail_ids.append(id)
                
                continue


    for pid, chunks in chunk_store.items():
        preds = []
        max_probs_all = []
        all_probs_all = []
        preds12_all = []
        all_probs12_all = []

        for idx in sorted(chunks):

            pred, max_prob, all_prob, pred12, all_prob12 = chunks[idx]
            preds.append(pred)
            if max_prob is not None:
                max_probs_all.append(max_prob)
            if all_prob is not None:
                all_probs_all.append(all_prob)
            if pred12 is not None:
                preds12_all.append(pred12)
            if all_prob12 is not None:
                all_probs12_all.append(all_prob12)

        # always needed to calculate the mean confidence
        pred_full = np.concatenate(preds)
        max_probs_full = np.concatenate(max_probs_all)
        mean_prob_full = round(100 * max_probs_full.mean(), 2)
        # absent unless --profiles kept the per-position distributions
        all_probs_full = np.concatenate(all_probs_all) if all_probs_all else None
        # only if every chunk carried a 12-state prediction, so the reassembled
        # 12-state string cannot end up shorter than the 3Di one
        preds12_full = (np.concatenate(preds12_all)
                        if len(preds12_all) == len(chunks) else None)
        all_probs12_full = (np.concatenate(all_probs12_all)
                            if len(all_probs12_all) == len(chunks) else None)

        batch_predictions[pid] = (
            pred_full,
            mean_prob_full,
            max_probs_full,
            all_probs_full,
            preds12_full,
            all_probs12_full
        )

    # Sort back in order of the original dictionary
    for k in original_keys:
        if k in batch_predictions:
            predictions[k] = batch_predictions[k]


    # write list of fails if length > 0
    if len(fail_ids) > 0:
        fail_tsv: Path = Path(output_dir) / "fails.tsv"

        # Convert the list to a list of lists
        data_as_list_of_lists = [[str(item)] for item in fail_ids]

        # Write the list to a TSV file
        with open(fail_tsv, "w", newline="") as file:
            tsv_writer = csv.writer(file, delimiter="\t")
            tsv_writer.writerows(data_as_list_of_lists)

    output_3di: Path = Path(output_dir) / "input_3di.fasta"
    output_12st: Path = Path(output_dir) / "input_12st.fasta"

    write_predictions(predictions, output_3di, output_12st)

    if profiles:

        print("You used --profiles")
        prof_db = f"{output_dir}/query_profiledb/query_profile"
        print(f"This will generate 3Di PSSM Foldseek profiles saved to {prof_db}")
        print("This will be used for the Foldseek searches")
        print(f"Note that while {output_3di} will be written, we do not recommend its use compared to the discrete 3Di predictions")
        print(f"Please use the default (without --profiles) if you want the best 3Di predictions in {output_3di}")


        lookup = pssm_db_builder["lookup"]
        parts = pssm_db_builder["parts"]
        index_lines = pssm_db_builder["index_lines"]
        offset = pssm_db_builder["offset"]


        # 12-state profile records, written as a sibling _ss12 database below
        ss12_entries = []

        for identifier, (pred, mean_prob, max_prob, all_prob, pred12, all_prob12) in predictions.items():

            # all_prob is  # (L, 20)

            logPSSM, consensus = computeLogPSSM(all_prob)

            key = lookup[identifier]            # mmseqs numeric id

            buf = toBuffer_pssm(logPSSM, consensus) + PROFILE_RECORD_END
            buf_length = len(buf)

            parts.append(buf)
            index_lines.append(f"{key}\t{offset}\t{buf_length}")
            offset += buf_length

            # update offset back into builder
            # pssm_db_builder["offset"] = offset # dont think this is needed as i am doing all at the end

            if all_prob12 is not None:
                # all_prob12 is (L, 12); its own background frequencies
                logPSSM12, consensus12 = computeLogPSSM(all_prob12, BACKPROBS_12ST)
                ss12_entries.append((key, toBuffer_pssm_states(
                    logPSSM12, consensus12, len(BACKPROBS_12ST))))

        # write pssm AFTER ALL

        db_buffer = b"".join(pssm_db_builder["parts"])
        pssm_output_db = pssm_db_builder["output_db"]      
        pssm_index = pssm_db_builder["index_file"]         

        # Single write for PSSM DB
        with open(pssm_output_db, "wb") as f:
            f.write(db_buffer)

        # Single write for PSSM index
        with open(pssm_index, "w") as f:
            f.write("\n".join(pssm_db_builder["index_lines"]) + "\n")

        # --- the 12-state profile channel, as Foldseek's <querydb>_ss12 ---
        if ss12_entries:
            ss12_db = f"{pssm_db_builder['seq_output_db']}_ss12"
            ss12_entries.sort(key=lambda kv: int(kv[0]))
            write_ss12_profile_db(ss12_entries, ss12_db)
            print(f"Finished writing 12-state PSSM profile to '{ss12_db}'")

        # --- ALSO build the amino-acid profile DB (seq_to_db) here ---

        mmseqs_db = pssm_db_builder["dummy_seq_db"]
        mmseqs_db_index = f"{mmseqs_db}.index"

        build_database_seq(mmseqs_db, 
                            mmseqs_db_index,
                            output_db=pssm_db_builder["seq_output_db"],
                            output_index=pssm_db_builder["seq_index_file"])

        # Copy extra files (.h, .dbtype, .lookup) and sort indices, same as script
        copy_and_create_extras(mmseqs_db)
        sort_index_file(pssm_index)

        # clean up teh dummy


        remove_file(mmseqs_db)
        remove_file(f"{mmseqs_db}.dbtype")
        remove_file(f"{mmseqs_db}_h")
        remove_file(f"{mmseqs_db}_h.dbtype")
        remove_file(f"{mmseqs_db}_h.index")
        remove_file(f"{mmseqs_db}.index")


        print(f"Finished writing 3di PSSM Foldseek profile to '{pssm_db_builder['seq_output_db']}'")

    # the 12-state FASTA only exists when the model has a 12st head
    return output_3di, (output_12st if Path(output_12st).exists() else None)

    
