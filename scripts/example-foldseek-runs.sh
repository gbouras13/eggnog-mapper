#!/usr/bin/env bash
# Three comparable runs over one query sample. Set QUERIES and DATA_DIR.
#
# Three configurations:
#   1. mmseqs                     - sequence-search baseline
#   2. mmseqs-foldseek            - + Foldseek on the leftovers, packed 3Di+12st
#   3. mmseqs-foldseek --profiles - + Foldseek on the leftovers, PSSM + _ss12
# Sequential on purpose: they would otherwise fight over the same 16 cores.
set -u
R=${OUTDIR:-./runs}
Q=${QUERIES:?set QUERIES=/path/to/queries.faa}
DD=${DATA_DIR:?set DATA_DIR=/path/to/db}
MMDB=${MMSEQS_DB:-$DATA_DIR/mmseqs/mmseqs.db}
FSDB=${FOLDSEEK_DB:-$DATA_DIR/foldseek_db/foldseek.db}
HF=${HF_CACHE:-$DATA_DIR/foldseek}

# single sensitivity pass at 7 instead of the 3->5->7 ladder
SENS="--start_sens 7 --sens_steps 1 --final_sens 7"

run () {
  name="$1"; shift
  out="$R/$name"; rm -rf "$out"; mkdir -p "$out"
  echo "=== [$name] start $(date +%T) ==="
  /usr/bin/time -f "[$name] wall %E maxRSS %MkB" \
  python -u emapper.py \
      --data_dir $DD --mmseqs_db $MMDB --hf_cache_dir $HF \
      -i "$Q" --itype proteins $SENS \
      --report_dropped --cpu 16 --override \
      -o "$name" --output_dir "$out" --temp_dir "$out" \
      "$@" > "$out/run.log" 2>&1
  rc=$?
  echo "=== [$name] exit $rc at $(date +%T) ==="
  tail -2 "$out/run.log" | tr '\r' '\n' | grep -vE '^\s*$' | sed 's/^/    /'
}

run mmseqs                 -m mmseqs
run mmseqs-foldseek        -m mmseqs-foldseek --foldseek_db $FSDB
run mmseqs-foldseek-prof   -m mmseqs-foldseek --foldseek_db $FSDB --profiles

echo "=== ALL DONE $(date +%T) ==="
