"""check_model_download honours --modernprost_revision.

Without a revision any complete cached snapshot is used. With one, only that
snapshot counts, so an older cached copy of the repo is not silently used
after the repo was updated.
"""

import sys

from tests.unit.test_foldseek_thresholds import _import_foldseek_module

_import_foldseek_module()
inference = sys.modules["eggnogmapper.search.foldseek.inference"]

REPO = "gbouras13/modernprost-50M-pssm"
OLD, NEW = "a" * 40, "b" * 40


def _cache(tmp_path, commits, refs):
    root = tmp_path / ("models--" + REPO.replace("/", "--"))
    for commit in commits:
        snap = root / "snapshots" / commit
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        (snap / "model.safetensors").write_text("")
    (root / "refs").mkdir(parents=True)
    for name, commit in refs.items():
        (root / "refs" / name).write_text(commit)
    return tmp_path


def test_any_snapshot_without_revision(tmp_path):
    cache = _cache(tmp_path, [OLD], {"main": OLD})
    assert inference.check_model_download(cache, REPO) is False


def test_pinned_commit_must_be_cached(tmp_path):
    cache = _cache(tmp_path, [OLD], {"main": OLD})
    assert inference.check_model_download(cache, REPO, NEW) is True
    assert inference.check_model_download(cache, REPO, OLD) is False


def test_pinned_branch_resolves_through_refs(tmp_path):
    cache = _cache(tmp_path, [OLD, NEW], {"main": OLD, "release": NEW})
    assert inference.check_model_download(cache, REPO, "release") is False
    assert inference.check_model_download(cache, REPO, "jul26") is True
