"""MMseqs2Searcher search-filter handling.

Named "filters" rather than "thresholds" deliberately: .gitignore carries a
`*old*` pattern, which matches "thresh*old*s" and would silently drop this
file from `git add`.

Covers two defects:

* ``--evalue`` was never passed to ``mmseqs search``, only applied as a filter
  on the converted m8. MMseqs2's own ``-e`` defaults to 0.001, which equals
  emapper's ``--evalue`` default, so the two agreed by coincidence -- but any
  looser threshold was silently capped at 0.001 because MMseqs2 discarded
  those alignments internally.
* ``_parse_mmseqs`` skipped a query as soon as it had *seen* it, before the
  thresholds were applied, so a query whose first candidate failed produced no
  seed even when a later candidate would have passed.
"""

import pytest

from eggnogmapper.search.mmseqs.mmseqs import MMseqs2Searcher


class _Searcher:
    """An MMseqs2Searcher carrying only the attributes the code under test reads.

    Built without __init__ so no parsed argparse Namespace, MMseqs2 binary or
    target database is needed.
    """

    def __new__(cls, **attrs):
        obj = MMseqs2Searcher.__new__(MMseqs2Searcher)
        defaults = dict(
            pident_thr=None, evalue_thr=None, score_thr=None,
            query_cov=None, subject_cov=None,
            cpu=1, temp_dir="/tmp", gpu=False,
            start_sens=3, sens_steps=3, final_sens=7,
            sub_mat=None, translation_table=None,
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
    import eggnogmapper.search.mmseqs.mmseqs as mod

    captured = {}

    class _Completed:
        stderr = b""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Completed()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    searcher.search_step("q.db", "t.db", "r.db")
    return captured["cmd"]


def test_evalue_is_passed_to_mmseqs_search(monkeypatch):
    cmd = _build_search_cmd(_Searcher(evalue_thr=1e-10), monkeypatch)
    assert " -e 1e-10" in cmd


def test_evalue_looser_than_mmseqs_default_reaches_the_search(monkeypatch):
    # The regression: MMseqs2's own -e default is 0.001, so anything looser was
    # discarded inside the search and could not be recovered by the post-filter.
    cmd = _build_search_cmd(_Searcher(evalue_thr=1.0), monkeypatch)
    assert " -e 1.0" in cmd


def test_no_evalue_flag_when_threshold_unset(monkeypatch):
    cmd = _build_search_cmd(_Searcher(evalue_thr=None), monkeypatch)
    assert " -e " not in cmd


def test_evalue_does_not_disturb_other_search_options(monkeypatch):
    cmd = _build_search_cmd(
        _Searcher(evalue_thr=1e-5, sub_mat="blosum62.out", translation_table=11),
        monkeypatch)
    assert " -e 1e-05" in cmd
    assert "--sub-mat blosum62.out" in cmd
    assert "--translation-table 11" in cmd
    assert "-s 7" in cmd


# --------------------------------------------------------------------------
# _parse_mmseqs: thresholds decide, then one seed per query
# --------------------------------------------------------------------------

def test_later_candidate_used_when_first_fails_evalue(tmp_path):
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "bad",  30.0, 1.0,   900),   # e=1.0  fails e<=1e-3
        ("q1", "good", 25.0, 1e-20, 850),   # e=1e-20 passes
    ]))

    hits = list(_Searcher(evalue_thr=1e-3)._parse_mmseqs(str(path)))

    assert len(hits) == 1
    assert hits[0][1] == "good"


def test_still_one_seed_per_query(tmp_path):
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "first",  30.0, 1e-30, 900),
        ("q1", "second", 29.0, 1e-25, 890),
        ("q2", "other",  40.0, 1e-40, 950),
    ]))

    hits = list(_Searcher(evalue_thr=1e-3)._parse_mmseqs(str(path)))

    assert [h[0] for h in hits] == ["q1", "q2"]
    assert [h[1] for h in hits] == ["first", "other"]


def test_query_dropped_when_every_candidate_fails(tmp_path):
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "bad1", 30.0, 1.0, 900),
        ("q1", "bad2", 29.0, 2.0, 890),
        ("q2", "ok",   40.0, 1e-40, 950),
    ]))

    hits = list(_Searcher(evalue_thr=1e-3)._parse_mmseqs(str(path)))

    assert [h[0] for h in hits] == ["q2"]


def test_pident_threshold_also_falls_through(tmp_path):
    """Independent of hit ordering, so this one could never be argued away."""
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "low_id",  10.0, 1e-30, 900),   # fails pident >= 20
        ("q1", "high_id", 80.0, 1e-25, 890),
    ]))

    hits = list(_Searcher(pident_thr=20.0)._parse_mmseqs(str(path)))

    assert len(hits) == 1
    assert hits[0][1] == "high_id"


def test_no_thresholds_keeps_first_hit_per_query(tmp_path):
    """Unfiltered behaviour is unchanged: first hit per query wins."""
    path = tmp_path / "hits.m8"
    path.write_text(_m8([
        ("q1", "a", 30.0, 5.0, 900),
        ("q1", "b", 29.0, 6.0, 890),
        ("q2", "c", 20.0, 7.0, 800),
    ]))

    hits = list(_Searcher()._parse_mmseqs(str(path)))

    assert [h[1] for h in hits] == ["a", "c"]
