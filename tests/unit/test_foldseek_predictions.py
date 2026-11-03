"""--foldseek_predictions: reusing an earlier run's ModernProst outputs."""

import sys

import pytest

from tests.unit.test_foldseek_thresholds import _import_foldseek_module

_import_foldseek_module()
inference = sys.modules["eggnogmapper.search.foldseek.inference"]

from eggnogmapper.emapperException import EmapperException  # noqa: E402


def _fasta(path, ids):
    path.write_text("".join(f">{i} desc\nMKT\n" for i in ids))
    return path


def _preds(d, ids, plddt=True, fails=()):
    d.mkdir()
    _fasta(d / "input_3di.fasta", ids)
    _fasta(d / "input_12st.fasta", ids)
    if plddt:
        (d / "input_plddt.tsv").write_text("".join(f"{i}\t90.0,90.0,90.0\n" for i in ids))
    if fails:
        (d / "fails.tsv").write_text("".join(f"{i}\n" for i in fails))
    return d


def test_reuses_the_three_files(tmp_path):
    fasta = _fasta(tmp_path / "q.faa", ["a", "b"])
    d = _preds(tmp_path / "pred", ["b", "a"])
    out_3di, out_12st, plddt = inference.load_predictions(d, fasta, need_plddt=True)
    assert (out_3di, out_12st, plddt) == (d / "input_3di.fasta", d / "input_12st.fasta",
                                         d / "input_plddt.tsv")
    assert inference.load_predictions(d, fasta, need_plddt=False)[2] is None


def test_queries_listed_as_failed_may_be_missing(tmp_path):
    fasta = _fasta(tmp_path / "q.faa", ["a", "b"])
    d = _preds(tmp_path / "pred", ["a"], fails=["b"])
    assert inference.load_predictions(d, fasta, need_plddt=True)[0] == d / "input_3di.fasta"


@pytest.mark.parametrize("pred_ids", [["a"], ["a", "b", "c"]])
def test_predictions_for_another_input_are_rejected(tmp_path, pred_ids):
    fasta = _fasta(tmp_path / "q.faa", ["a", "b"])
    d = _preds(tmp_path / "pred", pred_ids)
    with pytest.raises(EmapperException, match="do not match"):
        inference.load_predictions(d, fasta, need_plddt=False)


def test_masking_needs_the_plddt_file(tmp_path):
    fasta = _fasta(tmp_path / "q.faa", ["a"])
    d = _preds(tmp_path / "pred", ["a"], plddt=False)
    with pytest.raises(EmapperException, match="input_plddt.tsv"):
        inference.load_predictions(d, fasta, need_plddt=True)
