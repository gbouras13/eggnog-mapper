#!/usr/bin/env python3
"""Build a 12-state Foldseek target database from protein sequences.

eggnog-mapper does not ship a Foldseek target DB and download_eggnog_data.py
does not fetch one, so the Foldseek search modes need one built locally.

A Foldseek target database has the same on-disk layout as the query database
eggnog-mapper builds, so one can be made without any structures: predict 3Di +
12-state with ModernProst and pack both into _ss (one byte per residue,
3Di * 12 + 12st, dbtype flagged with DBTYPE_EXTENDED_3DI_12ST).

Two requirements for the result to be usable:

  * Targets must be named like eggNOG members (e.g. 362663.ECP_0061), so the
    annotation step can resolve the seed ortholog.
  * The same ModernProst checkpoint must be used here and for the queries
    searched against it (--model / emapper's --modernprost_model). Both
    alphabets are model-specific; mixing checkpoints silently degrades scores.

Cost scales with the number of targets: roughly 120-160 sequences/s on one
modern GPU, so ~15 min for 100k and ~50 h for a full 21.8M-protein eggNOG set.
A taxonomically scoped subset is usually the sensible unit of work.

Usage:
    # from a protein FASTA
    scripts/build-foldseek-db.py targets.faa OUTDIR/foldseek.db

    # or pull the proteins out of an existing MMseqs/Foldseek sequence DB
    scripts/build-foldseek-db.py --from-seqdb data/mmseqs/mmseqs.db \
        OUTDIR/foldseek.db
"""
import argparse
import contextlib
import io
import os
import sys
import tempfile


def seqdb_to_fasta(db_prefix, out_fasta):
    """Dump an MMseqs/Foldseek sequence DB (data + .index + _h) to FASTA."""
    def read(base):
        data = open(base, "rb").read()
        rows = []
        for line in open(base + ".index").read().strip().split("\n"):
            k, off, ln = line.split("\t")
            rows.append((k, data[int(off):int(off) + int(ln)]))
        return rows

    seqs = read(db_prefix)
    headers = dict(read(db_prefix + "_h"))
    with open(out_fasta, "w") as fh:
        for key, seq in seqs:
            name = headers[key][:-2].decode().split()[0]
            fh.write(f">{name}\n{seq[:-2].decode()}\n")
    return len(seqs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasta", nargs="?", help="protein FASTA of target sequences")
    ap.add_argument("out_db", help="output DB prefix, e.g. DIR/foldseek.db")
    ap.add_argument("--from-seqdb", metavar="PREFIX",
                    help="take the targets from this MMseqs/Foldseek sequence DB instead")
    ap.add_argument("--emapper-root", default=os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))),
                    help="path to the eggnog-mapper checkout")
    ap.add_argument("--model", default=None,
                    help="ModernProst checkpoint (default: the same one emapper uses)")
    ap.add_argument("--hf-cache", default=os.environ.get("HF_HOME"),
                    help="Hugging Face cache dir for the ModernProst weights")
    ap.add_argument("--foldseek", default="foldseek",
                    help="foldseek binary to use for tsv2db")
    ap.add_argument("--no-12st", action="store_true",
                    help="write a legacy 3Di-only _ss instead")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batch-residues", type=int, default=50000)
    args = ap.parse_args()

    if not args.fasta and not args.from_seqdb:
        ap.error("give a FASTA or --from-seqdb")

    sys.path.insert(0, args.emapper_root)
    from eggnogmapper.search.foldseek import inference as inf
    from eggnogmapper.search.foldseek import create_fs_db as cfd
    cfd.FOLDSEEK = args.foldseek

    out_dir = os.path.dirname(os.path.abspath(args.out_db))
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="fs_testdb_", dir=out_dir)

    fasta = args.fasta
    if args.from_seqdb:
        fasta = os.path.join(work, "targets.faa")
        n = seqdb_to_fasta(args.from_seqdb, fasta)
        print(f"extracted {n} sequences from {args.from_seqdb}")

    names = [l[1:].split()[0] for l in open(fasta) if l.startswith(">")]
    print(f"predicting structural alphabets for {len(names)} targets")

    cache = args.hf_cache or os.path.join(work, "hf")
    with contextlib.redirect_stdout(io.StringIO()):
        kw = {"model_name": args.model} if args.model else {}
        out_3di, out_12st = inf.infer_3Di(fasta, work, cache, args.batch_size,
                                          args.max_batch_residues, profiles=False, **kw)
    if args.no_12st:
        out_12st = None
    elif out_12st is None:
        sys.exit("model produced no 12-state head; re-run with --no-12st")

    with contextlib.redirect_stdout(io.StringIO()):
        cfd.generate_foldseek_db_from_aa_3di(fasta, out_3di, work, args.out_db,
                                             fasta_12st=out_12st)

    has12 = cfd.db_has_12st(args.out_db)
    print(f"wrote {args.out_db}  (12-state: {has12})")
    print("targets:", ", ".join(names[:8]) + (" ..." if len(names) > 8 else ""))
    print(f"\nuse it with:\n  emapper.py -m foldseek --foldseek_db {args.out_db} \\\n"
          f"      -i QUERIES.faa --itype proteins -o test --output_dir OUT")


if __name__ == "__main__":
    main()
