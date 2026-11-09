##
## Gbouras 2026

# Masking low-confidence query residues by predicted pLDDT.
#
# The ModernProst repos ship an optional per-residue pLDDT head in a
# plddt_head/ subfolder (plddt_head.py, head_config.json, head.safetensors).
# It reads the frozen backbone's encoder layers, so pLDDT comes out of the same
# forward pass as the structural alphabets. Residues predicted below the
# threshold are masked in the query's structural channels. Profile queries
# (--profiles) are lowercased in a "marks" amino-acid database keyed like the
# query database, and `foldseek lowconfquery` zeroes their 3Di (_ss) and
# 12-state (_ss12) columns. Discrete queries get their packed 3Di+12-state
# residue set to X here (mask_packed_query_db). The amino-acid channel is not
# masked.

import importlib.util
import os
import shutil
import subprocess
import struct
from pathlib import Path

from ...common import FOLDSEEK
from ...emapperException import EmapperException
from .profiles import PROFILE_RECORD_END

PLDDT_HEAD_SUBFOLDER = "plddt_head"
# Predicted-pLDDT threshold applied to Foldseek queries unless --plddt_mask
# says otherwise (0 turns masking off)
PLDDT_MASK_DEFAULT = 50.0
# MMseqs DBTYPE_AMINO_ACIDS
DBTYPE_AMINO_ACIDS = 0
# Foldseek Alphabet12St::PACKED_X: the packed 3Di+12-state byte read as X in
# both alphabets
PACKED_X = 252


def resolve_plddt_mask(plddt_mask, profiles, ss_12st=1):
    """The masking threshold to use, or None for no masking.

    plddt_mask is the --plddt_mask value (None when not given); it defaults to
    PLDDT_MASK_DEFAULT and 0 turns masking off. Discrete queries can only be
    masked in the packed 3Di+12-state form (Foldseek has a masked state only
    there), so with --ss_12st 0 the default is no masking and asking for a
    threshold is an error.
    """
    if plddt_mask is not None and plddt_mask < 0:
        raise EmapperException(f"--plddt_mask must be >= 0, not {plddt_mask:g}")
    if not profiles and not ss_12st:
        if plddt_mask:
            raise EmapperException(
                "--plddt_mask needs --profiles or the 12-state alphabet (--ss_12st 1)")
        return None
    if plddt_mask is None:
        plddt_mask = PLDDT_MASK_DEFAULT
    return plddt_mask if plddt_mask > 0 else None


def load_plddt_head(model_name, cache_dir, revision=None, device="cpu", local_files_only=False):
    """Load the pLDDT head shipped with a ModernProst model repo or local model dir.

    Returns (module, head): module.predict_plddt(model, head, input_ids,
    attention_mask) gives (pLDDT (B, L) in 0-100, the backbone's output).
    """
    local = Path(model_name) / PLDDT_HEAD_SUBFOLDER / "plddt_head.py"
    if local.is_file():
        code = str(local)
    else:
        from huggingface_hub import hf_hub_download
        kwargs = dict(subfolder=PLDDT_HEAD_SUBFOLDER, revision=revision, cache_dir=cache_dir)
        try:
            try:
                code = hf_hub_download(model_name, "plddt_head.py",
                                       local_files_only=local_files_only, **kwargs)
            except Exception:
                if not local_files_only:
                    raise
                # cached before the repo had a head: fetch just the head files
                code = hf_hub_download(model_name, "plddt_head.py", **kwargs)
        except Exception as e:
            raise EmapperException(
                f"{model_name} has no pLDDT head ({PLDDT_HEAD_SUBFOLDER}/plddt_head.py), "
                f"which --plddt_mask needs to mask low-confidence residues; use "
                f"--plddt_mask 0 to search without masking: {e}")

    spec = importlib.util.spec_from_file_location("modernprost_plddt_head", code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hub_kwargs = {} if local.is_file() else {"cache_dir": cache_dir}
    head = module.load_plddt_head(model_name, revision=revision, device=device, **hub_kwargs)
    return module, head


def write_plddt_tsv(plddts, out_tsv):
    """Write per-residue predicted pLDDT, one protein per line: id, comma-separated values."""
    with open(out_tsv, "w") as fh:
        for seq_id, values in plddts.items():
            fh.write(f"{seq_id}\t{','.join(f'{v:.1f}' for v in values)}\n")
    return


def read_plddt_tsv(in_tsv):
    """Read write_plddt_tsv output back: {id: [pLDDT, ...]}."""
    plddts = {}
    with open(in_tsv) as fh:
        for line in fh:
            seq_id, values = line.rstrip("\n").split("\t")
            plddts[seq_id] = [float(v) for v in values.split(",")] if values else []
    return plddts


def _read_db_records(db_prefix):
    """{key: record} of an MMseqs database, without the trailing newline and null byte."""
    with open(db_prefix, "rb") as fh:
        data = fh.read()
    records = {}
    with open(f"{db_prefix}.index") as fh:
        for line in fh:
            key, offset, length = line.split("\t")
            offset, length = int(offset), int(length)
            records[key] = data[offset:offset + length].rstrip(b"\x00").rstrip(b"\n")
    return records


def marks_from_querydb(querydb, plddts):
    """(key, sequence, pLDDT values) for every entry of a Foldseek query database.

    Sequences come from the database's amino-acid records and ids from its _h
    headers, so the marks line up with its keys and residues by construction.
    """
    seqs = _read_db_records(querydb)
    headers = _read_db_records(f"{querydb}_h")
    entries = []
    for key in sorted(seqs, key=int):
        seq_id = headers[key].decode().split()[0]
        if seq_id not in plddts:
            raise EmapperException(f"No predicted pLDDT for query {seq_id}")
        entries.append((key, seqs[key].decode(), plddts[seq_id]))
    return entries


def write_marks_db(entries, out_db, threshold):
    """Write the amino-acid "marks" database read by `foldseek lowconfquery`.

    entries: iterable of (key, sequence, pLDDT values) in key order, keyed like
    the query database. A residue is lowercased where its pLDDT is
    below threshold, which is what lowconfquery masks.

    Returns (masked residues, total residues).
    """
    parts, index_lines, offset = [], [], 0
    n_low = n_res = 0
    for key, seq, values in entries:
        seq = seq.upper()
        if len(seq) != len(values):
            raise EmapperException(
                f"Query {key}: {len(seq)} residues but {len(values)} pLDDT values")
        low = [v < threshold for v in values]
        marked = "".join(c.lower() if l else c for c, l in zip(seq, low))
        buf = (marked + "\n").encode() + PROFILE_RECORD_END
        parts.append(buf)
        index_lines.append(f"{key}\t{offset}\t{len(buf)}")
        offset += len(buf)
        n_low += sum(low)
        n_res += len(seq)

    with open(out_db, "wb") as fh:
        fh.write(b"".join(parts))
    with open(f"{out_db}.index", "w") as fh:
        if index_lines:
            fh.write("\n".join(index_lines) + "\n")
    with open(f"{out_db}.dbtype", "wb") as fh:
        fh.write(struct.pack("<i", DBTYPE_AMINO_ACIDS))
    return n_low, n_res


def mask_packed_query_db(query_db, entries, threshold):
    """Set the low-pLDDT residues of a packed 3Di+12-state query (_ss) to X, in place.

    entries: (key, sequence, pLDDT values) as from marks_from_querydb. This is
    what `foldseek lowconfquery` did to packed databases up to Foldseek
    583229dc; later Foldseek builds take only profile databases there, so it is
    done here, independently of the Foldseek version.

    Returns (masked residues, total residues).
    """
    ss_db = f"{query_db}_ss"
    records = _read_db_records(ss_db)
    n_low = n_res = 0
    for key, seq, values in entries:
        rec = bytearray(records[key])
        if len(rec) != len(values):
            raise EmapperException(
                f"Query {key}: {len(rec)} 3Di+12-state residues but {len(values)} pLDDT values")
        for i, v in enumerate(values):
            if v < threshold:
                rec[i] = PACKED_X
                n_low += 1
        n_res += len(rec)
        records[key] = bytes(rec)

    parts, index_lines, offset = [], [], 0
    for key in sorted(records, key=int):
        buf = records[key] + b"\n" + PROFILE_RECORD_END
        parts.append(buf)
        index_lines.append(f"{key}\t{offset}\t{len(buf)}")
        offset += len(buf)
    with open(ss_db, "wb") as fh:
        fh.write(b"".join(parts))
    with open(f"{ss_db}.index", "w") as fh:
        if index_lines:
            fh.write("\n".join(index_lines) + "\n")
    return n_low, n_res


def mask_query_db(query_db, marks_db, threads):
    """Zero the 3Di (_ss) and 12-state (_ss12) profile columns marked in marks_db, in place.

    Returns the commands run.
    """
    cmds = []
    for suffix in ("_ss", "_ss12"):
        db = f"{query_db}{suffix}"
        if not os.path.exists(f"{db}.index"):
            continue
        full = f"{db}_full"
        for ext in ("", ".index", ".dbtype"):
            shutil.move(f"{db}{ext}", f"{full}{ext}")

        cmd = (
            f'{FOLDSEEK} lowconfquery \'{marks_db}\' \'{full}\' \'{db}\' --threads {threads}'
        )
        try:
            subprocess.run(cmd, capture_output=True, check=True, shell=True)
        except subprocess.CalledProcessError as cpe:
            raise EmapperException("Error running 'foldseek lowconfquery': "
                                   + cpe.stderr.decode("utf-8").strip().split("\n")[-1])
        for ext in ("", ".index", ".dbtype"):
            os.remove(f"{full}{ext}")
        cmds.append(cmd)
    return cmds

## END
