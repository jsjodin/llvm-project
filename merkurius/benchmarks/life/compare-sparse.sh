#!/usr/bin/env bash
# Compares life-sparse.m, Life on sparse tiles in a GPU process, under
# merkc with life-sparse.c, the same tiles in C with a kernel a step and the
# bookkeeping on the host, and with dense life.c with uint8_t cells over the
# whole world, all at -O2 on the same GPU, taking turns so that none gets
# the quieter part of the machine's time. Each must give the same cells.
#
#   compare-sparse.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is gfx1030 by default; rounds, each a run of all three, 5 by
# default. LIFE_WORLD (8192), LIFE_TILE (64), LIFE_SOUP (256) and LIFE_STEPS
# (1000) pass through; the dense program gets LIFE_WORLD as LIFE_SIZE. The
# first round's sparse runs also write each step's active tiles and time to
# $out/mercury.trace and $out/c.trace, for a chart. On a machine that
# others share, run it through utils/gpu/with-gpus.py.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-gfx1030}"
rounds="${3:-5}"
out="${LIFE_OUT:-$(mktemp -d)}"
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_TARGET_OFFLOAD=MANDATORY
export LIFE_WORLD="${LIFE_WORLD:-8192}"
export LIFE_SOUP="${LIFE_SOUP:-256}"
export LIFE_STEPS="${LIFE_STEPS:-1000}"

"$build/bin/merkc" -O2 -fopenmp -fdo-loops -fprocess-declarations \
  --offload-arch="$arch" "${flags[@]}" "$here/life-sparse.m" -o "$out/mercury"
"$build/bin/clang" -O2 -fopenmp --offload-arch="$arch" "${flags[@]}" \
  "$here/life-sparse.c" -o "$out/c"
"$build/bin/clang" -O2 -fopenmp --offload-arch="$arch" -DCELL=uint8_t \
  "${flags[@]}" "$here/life.c" -o "$out/dense"

: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for prog in mercury c dense; do
    if [[ $prog == dense ]]; then
      LIFE_SIZE="$LIFE_WORLD" "$out/$prog" > "$out/$prog.$r"
    elif ((r == 1)); then
      LIFE_TRACE=1 "$out/$prog" > "$out/$prog.$r"
      grep '^step ' "$out/$prog.$r" > "$out/$prog.trace"
    else
      "$out/$prog" > "$out/$prog.$r"
    fi
    # The sparse programs print min, max, average and bookkeeping; the dense
    # one Gcells/s, min, max and average.
    awk -v prog="$prog" -v r="$r" '
      /^Population:/ { pop = $2 }
      /^Checksum:/ { sum = $2 }
      /^Active tiles:/ { peak = $3; last = $6 }
      NF == 4 && $1 ~ /^[0-9.]+$/ {
        if (prog == "dense") print prog, r, $4, 0, pop, sum, "-", "-"
        else print prog, r, $3, $4, pop, sum, peak, last
      }
    ' "$out/$prog.$r" >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# The cells must agree; then each program's median over the rounds of its
# average step and of the bookkeeping's part of it.
awk '
  { avg[$1, ++n[$1]] = $3; keep[$1, n[$1]] = $4; cells[$5 " " $6] = 1
    if ($7 != "-") { peak[$1] = $7; last[$1] = $8 } }
  function median(a, p,    k, i, j, t, s) {
    k = n[p]
    for (i = 1; i <= k; i++) s[i] = a[p, i]
    for (i = 2; i <= k; i++) {
      t = s[i]
      for (j = i - 1; j >= 1 && s[j] > t; j--) s[j + 1] = s[j]
      s[j + 1] = t
    }
    return k % 2 ? s[(k + 1) / 2] : (s[k / 2] + s[k / 2 + 1]) / 2
  }
  END {
    m = 0
    for (c in cells) m++
    if (m != 1) { print "the programs gave different cells"; exit 1 }
    if (peak["mercury"] != peak["c"] || last["mercury"] != last["c"]) {
      print "the sparse programs had different tiles"; exit 1
    }
    printf "Active tiles: %s at most, %s at the end\n", peak["c"], last["c"]
    printf "%-34s %14s %16s\n", "Program", "Average (sec)", "Bookkeeping"
    split("mercury c dense", ps, " ")
    split("Mercury (GPU process, sparse);C (sparse);C (dense, uint8_t)", names, ";")
    for (i = 1; i <= 3; i++)
      printf "%-34s %14.6f %16.6f\n", names[i], median(avg, ps[i]), median(keep, ps[i])
  }' "$out/results"
echo "Traces: $out/mercury.trace, $out/c.trace" >&2
