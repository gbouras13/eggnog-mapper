"""FoldseekSearcher threshold handling.

Covers two defects fixed together:

* ``--evalue`` was never passed to ``foldseek search``. Foldseek's own default
  is 10 (it overrides MMseqs' 0.001), so any threshold above 10 was silently
  capped: the hits never reached the m8 that the post-filter reads.
* ``_parse_foldseek`` skipped a query as soon as it had *seen* it, before
  applying the thresholds. With ``--foldseek_top_hits > 1`` a query whose
  first candidate failed produced no seed at all, even when a later candidate
  passed.

``eggnogmapper.search.foldseek.foldseek`` imports the ModernProst inference
module at import time, which pulls in torch and transformers. Those are an
optional extra (``pip install eggnog-mapper[foldseek]``) and are not installed
for the normal test run, so they are stubbed here. Nothing under test touches
them -- the searcher only needs them for 3Di prediction, not for parsing.
"""

import sys
import types

import pytest


class _PermissiveModule(types.ModuleType):
    """Stub module that invents any attribute asked of it.

    inference.py touches torch at import time for type annotations
    (``torch.Tensor``) as well as for the names it imports. Returning ``object``
    keeps annotations valid without having to enumerate every attribute.
    """

    __path__ = []  # makes dotted submodule imports resolve

    def __getattr__(self, name):
        return object


def _import_foldseek_module():
    # inference.py does `import torch`, `import torch.nn.functional as F`,
    # `from transformers import AutoModel, AutoTokenizer` and
    # `from tqdm import tqdm`, all at module scope.
    for name in ("torch", "torch.nn", "torch.nn.functional",
                 "transformers", "tqdm"):
        sys.modules.setdefault(name, _PermissiveModule(name))

    # `import torch.nn.functional` resolves each level by attribute lookup on
    # its parent, and __getattr__ would answer those with `object`. Bind the
    # submodules as real attributes so the dotted import finds the modules.
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]

    # tqdm is called, not just referenced, so it needs to be callable.
    sys.modules["tqdm"].tqdm = lambda it=None, *a, **kw: it

    from eggnogmapper.search.foldseek import foldseek as fs
    return fs


fs = _import_foldseek_module()


class _Searcher:
    """A FoldseekSearcher with only the attributes the code under test reads.

    Built without __init__ so the test does not need a parsed argparse
    Namespace, a Foldseek binary or a target database.
    """

    def __new__(cls, **attrs):
        obj = fs.FoldseekSearcher.__new__(fs.FoldseekSearcher)
        defaults = dict(
            pident_thr=None, evalue_thr=None, score_thr=None,
            query_cov=None, subject_cov=None,
            cpu=1, temp_dir="/tmp", ss_12st=1, profiles=False,
            evalue_nn_mode=None, evalue_12st_profile_comp=None,
            use_reverse_score=1,
        )
        defaults.update(attrs)
        for k, v in defaults.items():
            setattr(obj, k, v)
        return obj


def _m8(rows):
    """query target pident alnlen qstart qend tstart tend evalue bits qcov tcov"""
    return "".join(
        "\t".join(str(c) for c in (
            q, t, pident, 100, 1, 100, 1, 100, evalue, bits, 0.9, 0.9)) + "\n"
        for q, t, pident, evalue, bits in rows
    )


# --------------------------------------------------------------------------
# -e reaches the search command
# --------------------------------------------------------------------------

def _build_search_cmd(searcher, monkeypatch):
    """Capture the command search_step would run, without running it."""
    captured = {}

    class _Completed:
        stderr = b""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Completed()

    monkeypatch.setattr(fs.subprocess, "run", fake_run)
    searcher.search_step("q.db", "t.db", "r.db", 10000, False, 9.5, False)
    return captured["cmd"]


def test_evalue_is_passed_to_foldseek_search(monkeypatch):
    cmd = _build_search_cmd(_Searcher(evalue_thr=1e-10), monkeypatch)
    assert " -e 1e-10 " in cmd


def test_evalue_above_foldseek_default_reaches_the_search(monkeypatch):
    # The regression: Foldseek's own evalThr is 10, so a looser threshold used
    # to be discarded inside the search and could never be recovered by the
    # post-filter.
    cmd = _build_search_cmd(_Searcher(evalue_thr=100.0), monkeypatch)
    assert " -e 100.0 " in cmd


def test_no_evalue_flag_when_threshold_unset(monkeypatch):
    cmd = _build_search_cmd(_Searcher(evalue_thr=None), monkeypatch)
    assert " -e " not in cmd


# --------------------------------------------------------------------------
# _parse_foldseek: thresholds decide, then one seed per query
# --------------------------------------------------------------------------

def test_later_candidate_used_when_first_fails_evalue(tmp_path):
    """--foldseek_top_hits 2: the first hit fails --evalue, the second passes."""
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "bad",  30.0, 1.0,     900),   # e=1.0  fails e<=1e-3
        ("q1", "good", 25.0, 1e-20,   850),   # e=1e-20 passes
    ]))

    hits = list(_Searcher(evalue_thr=1e-3)._parse_foldseek(str(path)))

    assert len(hits) == 1
    assert hits[0][1] == "good"


def test_still_one_seed_per_query(tmp_path):
    """Passing candidates beyond the first are not emitted."""
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "first",  30.0, 1e-30, 900),
        ("q1", "second", 29.0, 1e-25, 890),
        ("q2", "other",  40.0, 1e-40, 950),
    ]))

    hits = list(_Searcher(evalue_thr=1e-3)._parse_foldseek(str(path)))

    assert [h[0] for h in hits] == ["q1", "q2"]
    assert [h[1] for h in hits] == ["first", "other"]


def test_query_dropped_when_every_candidate_fails(tmp_path):
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "bad1", 30.0, 1.0, 900),
        ("q1", "bad2", 29.0, 2.0, 890),
        ("q2", "ok",   40.0, 1e-40, 950),
    ]))

    hits = list(_Searcher(evalue_thr=1e-3)._parse_foldseek(str(path)))

    assert [h[0] for h in hits] == ["q2"]


def test_other_thresholds_also_fall_through_to_next_candidate(tmp_path):
    """The reordering is not E-value specific."""
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "low_id",  10.0, 1e-30, 900),   # fails pident >= 20
        ("q1", "high_id", 80.0, 1e-25, 890),
    ]))

    hits = list(_Searcher(pident_thr=20.0)._parse_foldseek(str(path)))

    assert len(hits) == 1
    assert hits[0][1] == "high_id"


def test_no_thresholds_keeps_first_hit_per_query(tmp_path):
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "a", 30.0, 5.0, 900),
        ("q1", "b", 29.0, 6.0, 890),
    ]))

    hits = list(_Searcher()._parse_foldseek(str(path)))

    assert [h[1] for h in hits] == ["a"]
