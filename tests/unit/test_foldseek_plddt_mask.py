"""--plddt_mask: the marks database read by `foldseek lowconfquery`."""

import pytest

from tests.unit.test_foldseek_thresholds import _import_foldseek_module

_import_foldseek_module()

from eggnogmapper.emapperException import EmapperException  # noqa: E402
from eggnogmapper.search.foldseek import plddt as pl  # noqa: E402


def _read_db(prefix):
    data = open(prefix, "rb").read()
    recs = {}
    for line in open(f"{prefix}.index"):
        key, off, n = line.split("\t")
        recs[key] = data[int(off):int(off) + int(n)]
    return recs


def test_marks_lowercase_residues_below_threshold(tmp_path):
    out = str(tmp_path / "query_profile_marks")
    n_low, n_res = pl.write_marks_db(
        [("1", "mktay", [90.0, 59.9, 60.0, 10.0, 75.0]), ("2", "MX", [30.0, 80.0])],
        out, 60)
    recs = _read_db(out)
    assert recs["1"] == b"MkTaY\n\0"
    assert recs["2"] == b"mX\n\0"
    assert (n_low, n_res) == (3, 7)
    assert open(f"{out}.dbtype", "rb").read() == b"\0\0\0\0"


def test_marks_reject_length_mismatch(tmp_path):
    with pytest.raises(EmapperException):
        pl.write_marks_db([("1", "MKT", [90.0, 80.0])], str(tmp_path / "m"), 60)




def test_mask_at_50_by_default():
    assert pl.resolve_plddt_mask(None, profiles=True) == 50.0
    assert pl.resolve_plddt_mask(None, profiles=False, ss_12st=1) == 50.0


def test_plddt_mask_overrides_and_zero_turns_it_off():
    assert pl.resolve_plddt_mask(60.0, profiles=True) == 60.0
    assert pl.resolve_plddt_mask(60.0, profiles=False, ss_12st=1) == 60.0
    assert pl.resolve_plddt_mask(0.0, profiles=True) is None
    assert pl.resolve_plddt_mask(0.0, profiles=False, ss_12st=1) is None


def test_3di_only_queries_cannot_be_masked():
    # only the packed 3Di+12-state alphabet has a masked state
    assert pl.resolve_plddt_mask(None, profiles=False, ss_12st=0) is None
    assert pl.resolve_plddt_mask(0.0, profiles=False, ss_12st=0) is None
    with pytest.raises(EmapperException, match="--ss_12st"):
        pl.resolve_plddt_mask(50.0, profiles=False, ss_12st=0)


def test_negative_plddt_mask_is_rejected():
    with pytest.raises(EmapperException):
        pl.resolve_plddt_mask(-1.0, profiles=True)


def _write_db(prefix, records):
    off = 0
    with open(prefix, "wb") as fd, open(f"{prefix}.index", "w") as fi:
        for key, rec in records:
            fd.write(rec)
            fi.write(f"{key}\t{off}\t{len(rec)}\n")
            off += len(rec)


def test_marks_follow_the_query_db_keys(tmp_path):
    querydb = str(tmp_path / "q")
    # keys 1..N in FASTA order, as generate_foldseek_db_from_aa_3di writes them
    _write_db(querydb, [("1", b"MKT\n\0"), ("2", b"AYX\n\0")])
    _write_db(f"{querydb}_h", [("1", b"protB some desc\n\0"), ("2", b"protA\n\0")])
    plddts = {"protA": [10.0, 90.0, 20.0], "protB": [80.0, 30.0, 70.0]}
    tsv = tmp_path / "input_plddt.tsv"
    pl.write_plddt_tsv(plddts, tsv)
    entries = pl.marks_from_querydb(querydb, pl.read_plddt_tsv(tsv))
    assert entries == [("1", "MKT", [80.0, 30.0, 70.0]), ("2", "AYX", [10.0, 90.0, 20.0])]
    out = str(tmp_path / "q_marks")
    assert pl.write_marks_db(entries, out, 50) == (3, 6)
    assert _read_db(out) == {"1": b"MkT\n\0", "2": b"aYx\n\0"}


def test_marks_need_a_plddt_for_every_query(tmp_path):
    querydb = str(tmp_path / "q")
    _write_db(querydb, [("1", b"MKT\n\0")])
    _write_db(f"{querydb}_h", [("1", b"protB\n\0")])
    with pytest.raises(EmapperException, match="protB"):
        pl.marks_from_querydb(querydb, {})


def test_packed_query_residues_below_threshold_become_x(tmp_path):
    querydb = str(tmp_path / "q")
    # packed 3Di+12-state bytes are 0..239, X is 252; records end "\n\0"
    _write_db(f"{querydb}_ss", [("1", bytes([5, 17, 200]) + b"\n\0"), ("2", bytes([0, 239]) + b"\n\0")])
    entries = [("1", "MKT", [80.0, 30.0, 49.9]), ("2", "AY", [50.0, 10.0])]
    assert pl.mask_packed_query_db(querydb, entries, 50) == (3, 5)
    assert _read_db(f"{querydb}_ss") == {"1": bytes([5, pl.PACKED_X, pl.PACKED_X]) + b"\n\0",
                                          "2": bytes([0, pl.PACKED_X]) + b"\n\0"}


def test_packed_query_mask_rejects_length_mismatch(tmp_path):
    querydb = str(tmp_path / "q")
    _write_db(f"{querydb}_ss", [("1", bytes([5, 17]) + b"\n\0")])
    with pytest.raises(EmapperException):
        pl.mask_packed_query_db(querydb, [("1", "MKT", [80.0, 30.0, 20.0])], 50)
