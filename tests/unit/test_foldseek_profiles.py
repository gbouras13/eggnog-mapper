"""Layout of the --profiles query databases written for Foldseek.

Three defects in the writer, fixed together:

* No record ended with the null byte Foldseek writes and expects
  (DBReader::getSeqLen reads (bytes - 1) / 25), so every query profile was read
  one residue short.
* The amino-acid profile skipped X residues, but Foldseek reads it with the 3Di
  profile's length, so after an X its columns were out of register with the
  3Di and 12-state ones.
* The query / consensus bytes of the amino-acid profile indexed an alphabet
  with an 'O' before 'P', so P..Y were off by one.
"""

import sys
import types

import numpy as np

sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
sys.modules["tqdm"].tqdm = getattr(sys.modules["tqdm"], "tqdm", lambda it=None, *a, **kw: it)

from eggnogmapper.search.foldseek import profiles as pr  # noqa: E402

FOLDSEEK_AA = "ACDEFGHIKLMNPQRSTVWYX"


def _read_db(prefix):
    data = open(prefix, "rb").read()
    recs = {}
    for line in open(f"{prefix}.index"):
        key, off, n = line.split("\t")
        recs[key] = data[int(off):int(off) + int(n)]
    return recs


def _blocks():
    _, sub_matrix = pr.parse_substitution_matrix_seq(pr.blosum62_str)
    return pr.precompute_sub_blocks(sub_matrix)


def test_aa_profile_letters_use_foldseek_indices():
    blocks = _blocks()
    for aa in FOLDSEEK_AA:
        assert blocks[aa][20] == blocks[aa][21] == FOLDSEEK_AA.index(aa)


def test_aa_profile_has_a_column_for_every_residue():
    blocks = _blocks()
    prof = pr.generate_profile_for_sequence("MKXBP", blocks)
    assert len(prof) == 5 * pr.PROFILE_READIN_SIZE
    # X and letters outside the alphabet get the BLOSUM62 X column
    x_col = blocks["X"]
    assert prof[50:75] == x_col and prof[75:100] == x_col
    assert np.all(np.frombuffer(x_col[:20], dtype=np.int8) == -4)


def test_aa_profile_db_records_end_with_null(tmp_path):
    seqdb = tmp_path / "query"
    seqs = {"1": "MKTAY", "2": "MXX"}
    with open(seqdb, "wb") as fd, open(f"{seqdb}.index", "w") as fi:
        off = 0
        for key, seq in seqs.items():
            rec = (seq + "\n").encode() + b"\0"
            fd.write(rec)
            fi.write(f"{key}\t{off}\t{len(rec)}\n")
            off += len(rec)
    out = tmp_path / "query_profile"
    pr.build_database_seq(str(seqdb), f"{seqdb}.index", str(out), f"{out}.index")
    recs = _read_db(str(out))
    for key, seq in seqs.items():
        assert len(recs[key]) == len(seq) * pr.PROFILE_READIN_SIZE + 1
        assert recs[key][-1:] == b"\0"


def test_12st_profile_db_records_end_with_null(tmp_path):
    L = 7
    probs = np.full((L, 12), 1 / 12, dtype=np.float32)
    logpssm, cons = pr.computeLogPSSM(probs, pr.BACKPROBS_12ST)
    rec = pr.toBuffer_pssm_states(logpssm, cons, 12)
    out = tmp_path / "query_profile_ss12"
    pr.write_ss12_profile_db([("1", rec)], str(out))
    recs = _read_db(str(out))
    assert len(recs["1"]) == L * pr.PROFILE_READIN_SIZE + 1
    assert recs["1"][-1:] == b"\0"
