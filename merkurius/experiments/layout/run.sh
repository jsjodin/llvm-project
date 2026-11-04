#!/usr/bin/env bash
# Builds the cell layout model (model.c) once per layout and runs every
# workload at sizes from L1 to DRAM, on one pinned core, five times each;
# writes every run to results.tsv and the medians to summary.tsv.
#
#   run.sh [cc] [core]
#
# cc is the C compiler, the build's clang by default; core is the CPU to
# pin to, 2 by default.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cc="${1:-$here/../../../../build-merk/bin/clang}"
core="${2:-2}"
out="${LAYOUT_OUT:-$here/out}"
mkdir -p "$out"
for l in A B C; do
  "$cc" -O2 -march=native -Wall -Wextra -DLAYOUT_$l "$here/model.c" -o "$out/model$l"
done
results="$out/results.tsv"
printf 'layout\trun\tworkload\telements\tbytes_per_elem\tbuild_ns\tsum_ns\tcopy_ns\tsum_copy_ns\tcheck\tcheck_copy\n' > "$results"
sizes="1000 16000 32000 256000 2000000 16000000"
for run in 1 2 3 4 5; do
  for l in A B C; do
    for w in nrev nrev-poly; do
      # nrev prints its own row: name, size, seconds, ns/cell, checksum.
      taskset -c "$core" "$out/model$l" "$w" | awk -v l=$l -v r=$run -F'\t' \
        '{ split($4, a, " "); printf "%s\t%s\t%s\t400x3001\t\t%s\t\t\t\t%s\t\n", l, r, $1, a[1], $5 }' >> "$results"
    done
    for w in list tree pairs; do
      for n in $sizes; do
        taskset -c "$core" "$out/model$l" "$w" "$n" | sed "s/^/$l\t$run\t/" >> "$results"
      done
    done
  done
done
python3 -I "$here/summarize.py" "$results" > "$out/summary.tsv"
cat "$out/summary.tsv"
