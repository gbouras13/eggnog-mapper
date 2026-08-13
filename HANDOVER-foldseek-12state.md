# Handover: Foldseek 12-state search in eggNOG-mapper v3

This branch merges the [eggnog-mapper-foldseek](https://github.com/gbouras13/eggnog-mapper-foldseek)
fork into eggNOG-mapper **v3**, and extends it to the **12-state structural
alphabet** added by the Foldseek build in `mvankem/foldseek`.

It also adds a route to annotate against a legacy **eggNOG v5** database,
because the v7 database v3 requires is not published yet.

Status: **the integration is verified; the scoring is not yet trustworthy.**
See [Open problems](#open-problems) before using any numbers.

---

## 1. What changed, and why

### 1.1 The fork merge

The fork was based on v2.1.13 and shared no git history with this repo (it was
re-initialised, not forked). The merge was done by grafting the fork's tree onto
the v2.1.13 commit that *is* on `main` (`8abf461`) so a real three-way merge base
exists, then hand-porting each integration hook onto v3. The graft is recorded as
a second parent of `f4dc9f7`, so future merges from the fork have a base.

Adapted rather than copied, because v3 changed the APIs the fork hooks into:

| v2.1.13 | v3 |
|---|---|
| bundled `eggnogmapper/bin/` binaries, `LOCAL_*` paths | `shutil.which` + `require_tool()` |
| `output_seeds(..., outfmt_short, ...)` | `outfmt_short` removed |
| `run_full_pipeline()` | `run()`; `annotate()` returns `(hits, resumed_ref)` |
| `run_generator() -> (n, time)` | same contract kept; ids collected via a `collect_ids` set |

Bugs fixed while porting (all present in the fork):

* The two-stage modes passed `predictor` to `search()`, which **ignores its
  `infile`** when a predictor is present — silently discarding the filtered
  no-hit FASTA.
* The stage switch renamed `genepred_*`, which `wrap_up()` uses to move gene
  prediction output.
* Stage-2 outputs were not in `_output_files`, so `--override` / `--resume` /
  `--scratch_dir` ignored them.
* The stage-1 searcher's temp dir leaked.
* `run_generator(key_fn=lambda x: x[0])` scooped whole hit rows into the
  annotated-id set: items are `(hit, annotation)` pairs, so `x[0]` is the hit
  list, not the query name.
* An f-string with nested double quotes in the `--profiles` path — a
  `SyntaxError` on the Python 3.9–3.11 that `setup.cfg` allows.
* `.gitignore`'s `*old*` pattern matches "f**old**seek", so the entire package
  was untracked in the fork.

Deliberate scope decisions:

* The two-stage modes now **require `--itype proteins` or `CDS`**. They identify
  leftover queries by diffing against input FASTA headers, which gene prediction
  invalidates (queries become `contig_0`, `contig_1`, …). Plain `-m foldseek`
  still supports genome input.
* `FoldseekSearcher` is imported lazily, so `torch`/`transformers` stay an
  optional extra (`pip install eggnog-mapper[foldseek]`) rather than a hard
  dependency of every run.
* The fork's `container/setonix/Dockerfile` was **not** carried over — it is
  site-specific, installs from the fork's GitHub with a token, and contains
  debug leftovers. Foldseek was added to `apptainer/eggnog-mapper.def` instead.

### 1.2 The 12-state alphabet

Foldseek stores the second alphabet in **two different on-disk forms**, and both
are implemented:

**Discrete queries** — packed into the `_ss` entry, one byte per residue:

```
byte = state3di * 12 + state12st          values 0..251
_ss dbtype = DBTYPE_AMINO_ACIDS | (DBTYPE_EXTENDED_3DI_12ST << 16)   -> [0,0,32,0]
alphabets: 3Di  ACDEFGHIKLMNPQRSTVWY(X=20 masked)   from data/mat3di.out
           12st ACDEFGHIKLMN (no X; index 6 = invalid) from data/mat12st.out
```

`foldseek tsv2db` **cannot** produce this: the packed bytes span 0..251 and
collide with TAB/NUL/newline in a TSV, and `--output-dbtype` cannot set the
extended bit. `create_fs_db.write_packed_ss_db()` writes the data/index/dbtype
files directly; entries are `<bytes> + '\n' + '\0'`.

**Profile queries (`--profiles`)** — a sibling `<querydb>_ss12` database of
dbtype `DBTYPE_HMM_PROFILE`. Foldseek picks it up automatically
(`structurealign.cpp`: `queryHas12St = query3Di12St || (q12StDbr != NULL)`).
MMseqs profile records are a fixed 25 bytes (`PROFILE_AA_SIZE` 20 score columns
+ 5 trailing) regardless of alphabet size, so the 12 states occupy slots 0..11
and slots 12..19 are padded with `-128`. Foldseek indexes a profile-layout
12-state matrix by **amino-acid slot** (`subMatAA.aa2num[rowLetter]`) — which is
the identity here, because `mat12st.out` and the AA matrix share the letter
order, so the 12 states are AA slots 0..11.

`ModernProst3Di12StModel` returns `logits` as a **dict**
`{"3di": (B,L,20), "12st": (B,L,12)}`, not a tensor. `split_logits()` handles
both and tolerates single-head checkpoints (12-state simply absent).

### 1.3 Annotating against an eggNOG v5 database

`scripts/v5_to_v7_db.py` rewrites a v5 `eggnog.db` into the v7 core schema:

```
og            -> ogs         (name kept as "cluster@taxid")
event         -> sp_events   (CSV-of-names sides -> delta-varint id BLOBs)
prots.name    -> protein_names + integer-keyed prots
prots.orthoindex -> event_index
ref_terms created empty (get_ref_term has no callers)
```

Two v7 columns have no v5 source: `sp_overlap` is computed faithfully as the
species-set Jaccard of the two sides; `ev_lca` is computed as the deepest common
ancestor of the event's species when `--taxa-db` is given, else falls back to the
v5 event level. Note `taxa.db` stores `track` **leaf→root**, contrary to
`LineageCache.tracks()`' docstring.

Two code changes were needed for v5-derived data:

* **Seed IDs.** v7 target DBs name sequences by integer protein ID, v5 ones by
  `TAXID.PROTEIN`. `batch_annotate` now resolves a non-integer seed through
  `protein_names` (new `AnnotDB.get_protein_id`) instead of dropping it. Drop
  reason renamed `non_integer_seed_id` -> `unresolvable_seed_id`.
* **OG names.** `_parse_ogs_string` required the v7 `cluster@taxid|clade` form
  and *silently skipped* v5's `cluster@taxid`, leaving `eggNOG_OGs`,
  `max_annot_lvl`, `COG_category` and `Description` empty while everything else
  looked fine. It now accepts both; v7 parsing is unchanged.

**This is v3's annotation logic over v5-derived data — not equivalent to
eggnog-mapper 2.x output.** v3 filters donors per seed by `ev_lca` ceiling and
`sp_overlap`, for which v5 holds no authoritative values.

---

## 2. New CLI options

| option | default | meaning |
|---|---|---|
| `-m foldseek` | | Foldseek over predicted 3Di (+12-state) |
| `-m diamond-foldseek`, `-m mmseqs-foldseek` | | sequence search, then Foldseek on the queries it left unannotated |
| `--ss_12st {0,1}` | 1 | use the 12-state alphabet |
| `--evalue_nn_mode {0,1,2}` | 2 when `--ss_12st` | E-value calibration network; see §4 |
| `--evalue_12st_profile_comp {0,1}` | Foldseek's 0 | mode-2 composition source for profile queries |
| `--use_reverse_score {0,1}` | 1 | subtract reverse-alignment score |
| `--modernprost_model` | `gbouras13/modernprost-50M-pssm` | checkpoint; **must match the target DB's** |
| `--foldseek_db` | `<data_dir>/foldseek/foldseek.db` | target DB |
| `--foldseek_cluster_db` | off | search the clustered DB |
| `--foldseek_top_hits` | 1 | structural hits kept per query |
| `--foldseek_max_seqs` | 10000 | Foldseek `--max-seqs` |
| `--foldseek_sensitivity` | 9.5 | Foldseek `-s` |
| `--profiles` | off | PSSM query profiles + `_ss12` |
| `--hf_cache_dir` | `<data_dir>/foldseek` | ModernProst weights cache |
| `--batch_size`, `--max_batch_residues` | 20, 50000 | inference batching |
| `--gpu` | off | GPU prefilter for MMseqs2/Foldseek (needs `*.gpu` padded DBs) |

Two-stage output layout — the stages never overwrite each other:

```
out.emapper.seed_orthologs                 stage 1
out.emapper.annotations
out.emapper.no_hits.fasta                  queries stage 1 did not annotate
out.foldseek.extra.emapper.seed_orthologs  stage 2
out.foldseek.extra.emapper.annotations
```

---

## 3. Setup

Foldseek must be a build providing `--ss-12st`; **bioconda's does not**. Use a
12-state-capable build and put it on `PATH` (or pass `--ss_12st 0`).

```bash
conda env create -f environment-foldseek.yml     # micromamba/mamba fine
conda activate eggnog-foldseek
pip install .
ln -s /path/to/12state/foldseek $CONDA_PREFIX/bin/foldseek
```

Verified on aarch64 + NVIDIA GB10 (sm_121) with torch 2.13.0+cu130,
transformers 5.14.1, Python 3.12. Adjust the torch index in the YAML for other
hardware (`cu130` / `cu128` / `cpu`).

Databases (none are in this archive):

```bash
# eggNOG v5 set - the v7 set is not published; both flags are required
python download_eggnog_data.py -y --no-verify --db_version 5.0.2 --data_dir DB

# convert to the v7 schema v3 reads
python scripts/v5_to_v7_db.py DB/eggnog.db DB/eggnog.db.v7 --taxa-db DB/eggnog.taxa.db

# MMseqs2 target DB from the v5 sequences (names must be eggNOG protein names)
diamond getseq -d DB/eggnog_proteins.dmnd -o v5prots.faa
mmseqs createdb v5prots.faa DB/mmseqs/mmseqs.db

# Foldseek 12-state target DB (GPU; ~15 min per 100k targets)
python scripts/build-foldseek-db.py SUBSET.faa DB/foldseek_db/foldseek.db
```

The target DB **must** carry the 12 states or Foldseek silently falls back to
3Di-only scoring; eggnog-mapper warns when it detects this. `<db>_ss.dbtype`
reads `[0,0,32,0]` when it does. An existing structure-derived DB can be
upgraded in place with `foldseek add12st` (needs `_ca` coordinates).

---

## 4. Open problems

**Do not treat the recall numbers as biological results.** Three runs over
20,000 queries (`scripts/example-foldseek-runs.sh`) gave:

| run | annotated | % of 20k |
|---|---|---|
| `-m mmseqs` (21.84M targets) | 688 | 3.4% |
| `-m mmseqs-foldseek` (100k targets) | 11,602 | 58.0% |
| `-m mmseqs-foldseek --profiles` | 19,965 | 99.8% |

99.8% of a novel-families query set matching a 100k subset is not plausible.
Two causes, one fixed:

1. **E-value calibration (fixed, `eead171`).** Foldseek defaults to
   `--evalue-nn-mode 0`, the *3Di-only* composition network. All three runs
   scored with the 12-state channel while calibrating without it. Mode 2 is the
   3Di+12-state network. Measured on 203 profile queries: bitscores identical
   (202/202 — the network only affects calibration), median E-value moves
   `2.51e-16 -> 2.98e-12`, a factor of **1.4e5**. `--evalue-12st-profile-comp 1`
   gives a further 4.9x (`-> 8.27e-10`) and is better founded for our profiles,
   since it recovers composition from the full distribution rather than counting
   only the consensus letter. **The three runs above predate this fix.**

2. **Profile diversity (open).** Our PSSMs come from a single-sequence model
   prediction, so every column looks perfectly conserved (`Neff` ≈ 1) and even
   with the best calibration 94% of probe queries still clear `e<1e-3`. The fix
   is pseudocount smoothing toward the background, as MMseqs `--pca/--pcb` does
   for MSA-derived profiles.

   Note: bit factor **8 is correct** and must not be changed —
   `EvalueNeuralNet::LEGACY_12ST_PROFILE_BIT_FACTOR = 8.0f` and
   `Sequence::scoreToProba = 2^(score/bitFactor) * pBack` are the exact inverse
   of our `log2(p/pBack) * 8`. Rescaling would break
   `--evalue-12st-profile-comp 1`. (An earlier analysis in this work compared
   profile scores against substitution-matrix diagonals and wrongly concluded
   the 12-state scale was 3.5x too hot; that comparison is invalid, because a
   matrix diagonal averages over observed substitutions while a profile column
   asserts a state.)

Also unresolved / not investigated: `--prefilter-mode`,
`--sort-by-structure-bits`, profile `--pca/--pcb`, and the `FOLDSEEK_*`
environment switches referenced in `StructureSearch.cpp`.

Independent of calibration, the run-2 comparison is not like-for-like: MMseqs2
searched 21.84M targets, Foldseek 100k. E-values scale with search space, and
Foldseek-stage median identity was 12.4% versus 32.4% for MMseqs2 — remote
structural matches, which is what structural search is for, but well below where
eggNOG's ortholog-based annotation transfer is safe.

**Suggested next steps:** pseudocount smoothing; default
`--evalue_12st_profile_comp 1`; re-run all three with mode 2; add a
`--profiles --ss_12st 0` control to separate the new profile channel from the
pre-existing 3Di profile behaviour.

---

## 5. What is verified

Against the real Foldseek binary and the real ModernProst checkpoint:

* Packed `_ss` output is **byte-identical** to a database Foldseek itself built
  from PDB structures, dbtype included.
* End to end from sequence: 6 globins each recover their own structure as top
  hit, self-hit bitscores ~30% above 3Di-only (726 -> 964 etc.).
* `_ss12` profile channel is used **causally**: with it, scores rise
  (1139 -> 1633); with it removed, `--ss-12st 1` reproduces 3Di-only scores
  exactly.
* A packed query against a 3Di-only target degrades gracefully to 3Di-only
  scores rather than corrupting.
* v5→v7 conversion: 75,371,789 events, 0 dropped, `ev_lca` computed for
  99.999%; annotation validated on the converted 38 GB database (40/40 queries,
  0 dropped, ~17/25 columns), with correct calls (ychF/COG0012 GTPase;
  *E. coli* polB/COG0417, EC 2.7.7.7, ko:K02336).
* Two-stage handoff: 20,000 − 689 = 19,311 leftovers, matching
  `no_hits.fasta` exactly, zero seed overlap between stages.
* Test suite 57 passed / 94 skipped, unchanged from `main`. The 94 skips are
  pre-existing and want a v7 sample DB at
  `$EGGNOG_DATA_ROOT/e7/sample/final/mapper/eggnog.db`.

Not verified: full-scale behaviour beyond 404k queries; recall quality (see §4).

---

## 6. Known pre-existing issues (not introduced here)

* **`emapperdb-7.0.0` is not published.** `downloads.eggnogdb.org/emapper/`
  carries only `emapperdb-5.0.2` and `novel_fams-1.0.1`, and **no** version has
  the `manifest.json` the downloader now requires — so the documented
  `download_eggnog_data.py` invocation cannot succeed as written. `scripts/
  release-db.sh` is what generates manifests and evidently has not been run
  against the public host.
* **`--db e5-test` is broken for installed packages**: `backends.py` resolves
  `tests/fixtures` against the package root, but `tests/` is not shipped in the
  wheel. Works only from a source checkout.
* `infer_3Di` loads the whole query FASTA into memory. Fine to a few hundred
  thousand sequences; a genome-scale target build needs streaming.
* `USAGE.md` still documents `-m novel_fams` and `-m cache`, which v3 removed.
