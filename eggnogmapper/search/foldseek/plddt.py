##
## Gbouras 2026

# Masking low-confidence residues of --profiles queries by predicted pLDDT.
#
# The ModernProst repos ship an optional per-residue pLDDT head in a
# plddt_head/ subfolder (plddt_head.py, head_config.json, head.safetensors).
# It reads the frozen backbone's encoder layers, so pLDDT comes out of the same
# forward pass as the profiles. Residues predicted below the threshold are
# lowercased in a "marks" amino-acid database keyed like the query profiles,
# and `foldseek lowconfquery` then zeroes their 3Di and 12-state profile
# columns. The amino-acid profile is not masked.

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
# MMseqs DBTYPE_AMINO_ACIDS
DBTYPE_AMINO_ACIDS = 0


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
                f"which --plddt_mask needs: {e}")

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


def write_marks_db(entries, out_db, threshold):
    """Write the amino-acid "marks" database read by `foldseek lowconfquery`.

    entries: iterable of (key, sequence, pLDDT values) in key order, keyed like
    the query profile databases. A residue is lowercased where its pLDDT is
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


def mask_query_profiles(profile_db, marks_db, threads):
    """Zero the 3Di (_ss) and 12-state (_ss12) profile columns marked in marks_db, in place.

    Returns the commands run.
    """
    cmds = []
    for suffix in ("_ss", "_ss12"):
        db = f"{profile_db}{suffix}"
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
