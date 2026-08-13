#!/usr/bin/env python3
"""Convert a legacy eggNOG v5 eggnog.db into the v7 core schema that v3 reads.

eggnog-mapper v3 annotates against a different schema than v2 did:

    v5:  og(og, level, nm, description, COG_categories)
         event(i, level, og, side1, side2)          -- sides are CSV of names
         prots(name, ogs, orthoindex, pname, gos, kegg_*)

    v7:  ogs(name, clust_name, level, nm, ns, description, cog_cat)
         sp_events(i, name, og, og_lca, ev_lca, sp_overlap, side1, side2)
                                                    -- sides are delta-varint
                                                       BLOBs of integer ids
         event_index(protein_id, events)            -- delta-varint BLOB
         protein_names(id, name, taxid)
         prots(id, pname, gos, kegg_*, ogs)
         ref_terms(type, term, symbol, descr)

This script rewrites the four v5 tables into the seven v7 core tables so the
v3 annotator will run. Two v7 columns have no v5 source and are reconstructed:

  sp_overlap  species-set Jaccard between side1 and side2, computed from the
              taxid prefixes of the member names. Faithful.

  ev_lca      the event's LCA taxid. v5 does not store it. With --taxa-db it
              is computed as the deepest common ancestor of every species in
              the event; otherwise it falls back to the v5 event level, which
              is what v2 effectively used as the scope.

IMPORTANT: the result is v3's annotation logic running over v5-derived data.
It is not equivalent to eggnog-mapper 2.x output on the same database — v3
filters donors per seed by ev_lca ceiling and sp_overlap, concepts v5 has no
authoritative values for. Treat it as a way to exercise the v3 pipeline on v5
data, not as a substitute for a real v7 build.

Usage:
    scripts/v5_to_v7_db.py V5_EGGNOG.DB OUT_EGGNOG.DB [--taxa-db eggnog.taxa.db]

    # quick smoke test on a slice
    scripts/v5_to_v7_db.py in.db out.db --limit-events 5000
"""
import argparse
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eggnogmapper.annotator.codec import encode_intlist
from eggnogmapper.annotator import schema as v7schema

BATCH = 20000


def log(msg):
    print(f"[v5->v7] {msg}", flush=True)


def species_of(name):
    """taxid prefix of a "TAXID.PROTEIN" name, as str; None if malformed."""
    tax, sep, _ = name.partition(".")
    return tax if sep and tax else None


class LcaResolver:
    """Deepest common ancestor of a set of species taxids.

    Backed by LineageCache's root->leaf tracks. Species with no known track
    are ignored; if nothing resolves, the caller's fallback is used.
    """

    def __init__(self, taxa_db_path):
        from eggnogmapper.annotator.lineage import LineageCache
        self._lc = LineageCache(taxa_db_path=taxa_db_path)
        self._anc = None      # species -> frozenset of its ancestors (incl. itself)
        self._depth = None    # taxid -> depth from root
        self._memo = {}       # frozenset(species) -> lca, for repeated species sets
        self.hits = 0
        self.misses = 0

    def _load(self):
        if self._anc is not None:
            return
        # taxa.db stores `track` leaf->root ("72274,1236,1224,2,131567,1")
        # despite LineageCache.tracks()' docstring claiming root->leaf.
        anc, depth = {}, {}
        for sp, track in self._lc.tracks():
            anc[sp] = frozenset(track)
            # reversed(track) is root->leaf, so the index is the depth
            for d, node in enumerate(reversed(track)):
                if node not in depth or d > depth[node]:
                    depth[node] = d
        self._anc, self._depth = anc, depth

    def lca(self, species, fallback):
        """Deepest taxon that is an ancestor of every resolvable species.

        Done with set intersection rather than a pairwise track walk: with ~8
        species per event and 75M events, the walk would dominate the whole
        conversion, whereas the intersection runs in C.
        """
        self._load()
        key = frozenset(species)
        got = self._memo.get(key)
        if got is not None:
            self.hits += 1
            return got

        sets = [self._anc[sp] for sp in key if sp in self._anc]
        if not sets:
            self.misses += 1
            return fallback
        common = sets[0] if len(sets) == 1 else frozenset.intersection(*sets)
        if not common:
            self.misses += 1
            return fallback
        depth = self._depth
        result = max(common, key=lambda t: depth.get(t, -1))
        self.hits += 1
        if len(self._memo) < 2_000_000:
            self._memo[key] = result
        return result


def create_schema(out):
    cur = out.cursor()
    for name, ddl, indexes in v7schema.CORE_TABLES:
        cur.execute(ddl)
        for ix in indexes:
            cur.execute(ix)
    out.commit()
    cur.close()


def build_protein_names(v5, out):
    """protein_names from every name in v5 prots; returns {name: id}."""
    log("pass 1/4: protein_names")
    cur = v5.execute("SELECT name FROM prots")
    ids = {}
    rows = []
    nxt = 1
    t0 = time.time()
    while True:
        chunk = cur.fetchmany(BATCH)
        if not chunk:
            break
        for (name,) in chunk:
            if name in ids:
                continue
            sp = species_of(name)
            ids[name] = nxt
            rows.append((nxt, name, int(sp) if sp and sp.isdigit() else 0))
            nxt += 1
        if len(rows) >= BATCH:
            out.executemany("INSERT INTO protein_names(id,name,taxid) VALUES (?,?,?)", rows)
            rows.clear()
    if rows:
        out.executemany("INSERT INTO protein_names(id,name,taxid) VALUES (?,?,?)", rows)
    out.commit()
    log(f"  {len(ids)} proteins in {time.time()-t0:.1f}s")
    return ids


V5_PROT_COLS = ["pname", "gos", "kegg_ko", "kegg_ec", "kegg_pathway", "kegg_module",
                "kegg_reaction", "kegg_rclass", "kegg_brite", "kegg_tc", "kegg_cazy",
                "kegg_cog", "kegg_disease", "kegg_go", "kegg_drug", "kegg_pubmed",
                "kegg_network", "bigg_reaction", "pfam", "ogs"]


def convert_prots(v5, out, ids):
    log("pass 2/4: prots")
    cols = ",".join(V5_PROT_COLS)
    ph = ",".join("?" * (1 + len(V5_PROT_COLS)))
    cur = v5.execute(f"SELECT name,{cols} FROM prots")
    n = 0
    rows = []
    while True:
        chunk = cur.fetchmany(BATCH)
        if not chunk:
            break
        for r in chunk:
            pid = ids.get(r[0])
            if pid is None:
                continue
            rows.append((pid,) + tuple(r[1:]))
            n += 1
        out.executemany(
            f"INSERT OR REPLACE INTO prots(id,{cols}) VALUES ({ph})", rows)
        rows.clear()
    out.commit()
    log(f"  {n} prots rows")


def convert_ogs(v5, out):
    """v5 og -> v7 ogs. name is "og@level", matching prots.ogs in v5."""
    log("pass 3/4: ogs")
    n = 0
    rows = []
    for og, level, nm, descr, cog in v5.execute(
            "SELECT og, level, nm, description, COG_categories FROM og"):
        rows.append((f"{og}@{level}", og, str(level), nm, None, descr, cog))
        n += 1
        if len(rows) >= BATCH:
            out.executemany("INSERT OR REPLACE INTO ogs"
                            "(name,clust_name,level,nm,ns,description,cog_cat) "
                            "VALUES (?,?,?,?,?,?,?)", rows)
            rows.clear()
    if rows:
        out.executemany("INSERT OR REPLACE INTO ogs"
                        "(name,clust_name,level,nm,ns,description,cog_cat) "
                        "VALUES (?,?,?,?,?,?,?)", rows)
    out.commit()
    log(f"  {n} OGs")


def convert_events(v5, out, ids, lca, limit=None):
    """v5 event -> v7 sp_events, plus event_index from prots.orthoindex."""
    log("pass 4/4: sp_events (+ event_index)")
    q = "SELECT i, level, og, side1, side2 FROM event"
    if limit:
        q += f" LIMIT {int(limit)}"
    cur = v5.execute(q)
    n = dropped = 0
    dropped_ids = set()
    rows = []
    t0 = time.time()
    while True:
        chunk = cur.fetchmany(BATCH)
        if not chunk:
            break
        for i, level, og, s1, s2 in chunk:
            n1 = [x for x in (s1 or "").split(",") if x]
            n2 = [x for x in (s2 or "").split(",") if x]
            i1 = [ids[x] for x in n1 if x in ids]
            i2 = [ids[x] for x in n2 if x in ids]
            n += 1
            if not i1 or not i2:
                # record only the rejects: on a complete v5 DB every member name
                # is in prots, so this stays near-empty. Tracking the *kept* ids
                # instead would mean a 75M-element Python set (~4 GB).
                dropped += 1
                dropped_ids.add(i)
                continue
            sp1 = {species_of(x) for x in n1} - {None}
            sp2 = {species_of(x) for x in n2} - {None}
            union = sp1 | sp2
            overlap = (len(sp1 & sp2) / len(union)) if union else 0.0
            ev_lca = lca.lca(union, str(level)) if lca else str(level)
            rows.append((i, og, f"{og}@{level}", str(level), ev_lca, overlap,
                         encode_intlist(i1), encode_intlist(i2)))
        if rows:
            out.executemany(
                "INSERT OR REPLACE INTO sp_events"
                "(i,name,og,og_lca,ev_lca,sp_overlap,side1,side2) "
                "VALUES (?,?,?,?,?,?,?,?)", rows)
            rows.clear()
    out.commit()
    log(f"  {n - dropped} events written, {dropped} dropped (empty side) "
        f"in {time.time()-t0:.1f}s")
    if lca:
        log(f"  ev_lca: {lca.hits} computed, {lca.misses} fell back to level")

    # event_index: protein -> its events, from v5 prots.orthoindex
    log("      event_index")
    cur = v5.execute("SELECT name, orthoindex FROM prots WHERE orthoindex IS NOT NULL")
    m = 0
    rows = []
    while True:
        chunk = cur.fetchmany(BATCH)
        if not chunk:
            break
        for name, oi in chunk:
            pid = ids.get(name)
            if pid is None or not oi:
                continue
            evs = [int(x) for x in oi.split(",") if x and x.isdigit()]
            if dropped_ids:
                evs = [e for e in evs if e not in dropped_ids]
            if not evs:
                continue
            rows.append((pid, encode_intlist(evs)))
            m += 1
        if rows:
            out.executemany(
                "INSERT OR REPLACE INTO event_index(protein_id,events) VALUES (?,?)",
                rows)
            rows.clear()
    out.commit()
    log(f"  {m} event_index rows")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("v5_db")
    ap.add_argument("out_db")
    ap.add_argument("--taxa-db", help="eggnog.taxa.db, enables computed ev_lca")
    ap.add_argument("--limit-events", type=int,
                    help="convert only the first N events (smoke test)")
    ap.add_argument("--force", action="store_true", help="overwrite OUT_DB")
    args = ap.parse_args()

    if os.path.exists(args.out_db):
        if not args.force:
            sys.exit(f"{args.out_db} exists; pass --force to overwrite")
        os.remove(args.out_db)

    v5 = sqlite3.connect(f"file:{args.v5_db}?mode=ro", uri=True)
    have = {r[0] for r in v5.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    missing = {"og", "event", "prots"} - have
    if missing:
        sys.exit(f"{args.v5_db} does not look like a v5 eggnog.db "
                 f"(missing tables: {sorted(missing)})")
    if "event_index" in have:
        sys.exit(f"{args.v5_db} already has event_index; it is not a v5 database")

    ver = v5.execute("SELECT version FROM version").fetchone()
    log(f"source version: {ver[0] if ver else 'unknown'}")

    out = sqlite3.connect(args.out_db)
    out.execute("PRAGMA journal_mode=OFF")
    out.execute("PRAGMA synchronous=OFF")
    create_schema(out)

    lca = LcaResolver(args.taxa_db) if args.taxa_db else None
    if lca is None:
        log("no --taxa-db: ev_lca will be set to the v5 event level")

    ids = build_protein_names(v5, out)
    convert_prots(v5, out, ids)
    convert_ogs(v5, out)
    convert_events(v5, out, ids, lca, args.limit_events)

    # v3 gates on event_index existing; keep the source version string so the
    # provenance of the converted DB is visible in --version output.
    out.execute("INSERT OR REPLACE INTO version(version) VALUES (?)",
                (f"{ver[0] if ver else 'v5'}-converted-v7",))
    out.commit()

    log("ANALYZE")
    out.execute("ANALYZE")
    out.commit()
    out.close()
    v5.close()
    log(f"wrote {args.out_db} "
        f"({os.path.getsize(args.out_db)/2**30:.2f} GB)")


if __name__ == "__main__":
    main()
