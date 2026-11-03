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



def test_profiles_mask_at_50_by_default():
    assert pl.resolve_plddt_mask(None, profiles=True) == 50.0


def test_plddt_mask_overrides_and_zero_turns_it_off():
    assert pl.resolve_plddt_mask(60.0, profiles=True) == 60.0
    assert pl.resolve_plddt_mask(0.0, profiles=True) is None


def test_no_masking_without_profiles():
    assert pl.resolve_plddt_mask(None, profiles=False) is None
    assert pl.resolve_plddt_mask(0.0, profiles=False) is None
    with pytest.raises(EmapperException, match="--profiles"):
        pl.resolve_plddt_mask(50.0, profiles=False)


def test_negative_plddt_mask_is_rejected():
    with pytest.raises(EmapperException):
        pl.resolve_plddt_mask(-1.0, profiles=True)
