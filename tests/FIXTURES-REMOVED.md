# Test fixtures are not in this archive

`tests/fixtures/` and `tests/fixtures_v7/` were removed to keep the archive to
source only — they are ~35 MB of binary databases (`eggnog.db`,
`eggnog_proteins.dmnd`, `Pfam-A.hmm`, HMMER indices, an MMseqs2 DB and a
taxonomy pickle).

Nothing in the currently-passing test suite needs them: `pytest tests/unit
eggnogmapper/annotator/tests` gives the same 57 passed / 94 skipped with or
without. The 94 skips are pre-existing and want a v7 sample database at
`$EGGNOG_DATA_ROOT/e7/sample/final/mapper/eggnog.db`.

What the fixtures do enable, if you want them back from the upstream repo
(`git checkout <rev> -- tests/fixtures tests/fixtures_v7`):

* `--db e5-test`, the tiny bundled v5 backend (source checkouts only — see the
  known-issues section of HANDOVER-foldseek-12state.md).
* `tests/integration/`, which needs real databases and search binaries.
