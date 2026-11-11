#!/usr/bin/env bash
# Compares front.m under merkc with front.c under the same build's clang,
# both at -O2 on the same GPU, taking turns so that neither gets the quieter
# part of the machine's time. They must burn the same cells, and their sums
# agree to 12 digits: clang fuses multiplies and adds on the GPU, merkc not.
#
#   compare-c.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is gfx1030 by default; rounds, each a run of both, 5 by
# default. FRONT_SIZE, FRONT_STEPS and FRONT_SUBSTEPS pass through. On a
# machine that others share, run it through utils/gpu/with-gpus.py.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-gfx1030}"
rounds="${3:-5}"
out="${FRONT_OUT:-$(mktemp -d)}"
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_TARGET_OFFLOAD=MANDATORY

"$build/bin/merkc" -O2 -fopenmp -fdo-loops --offload-arch="$arch" "${flags[@]}" \
  "$here/front.m" -o "$out/mercury"
"$build/bin/clang" -O2 -fopenmp --offload-arch="$arch" "${flags[@]}" \
  "$here/front.c" -o "$out/c"

: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for prog in mercury c; do
    "$out/$prog" > "$out/$prog.$r"
    awk -v prog="$prog" -v r="$r" '
      /^Burnt cells:/ { burnt = $3 }
      /^Heat:/ { heat = $2 }
      /^Fuel:/ { fuel = $2 }
      NF == 4 && $1 ~ /^[0-9.]+$/ { print prog, r, $1, $2, $4, burnt, heat, fuel }
    ' "$out/$prog.$r" >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# The cells must agree; then each program's median over the rounds of its
# fastest step and its average step.
awk '
  function far(a, b) { d = a - b; if (d < 0) d = -d; return d > 1e-12 * (a < 0 ? -a : a) }
  {
    rate[$1, ++n[$1]] = $3; avg[$1, n[$1]] = $5
    if (!seen) { burnt = $6; heat = $7; fuel = $8; seen = 1 }
    else if ($6 != burnt || far($7, heat) || far($8, fuel)) bad = 1
  }
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
    if (bad) { print "the programs gave different cells"; exit 1 }
    printf "%-10s %12s %16s\n", "Program", "Gcells/sec", "Average (sec)"
    printf "%-10s %12.3f %16.6f\n", "Mercury", median(rate, "mercury"), median(avg, "mercury")
    printf "%-10s %12.3f %16.6f\n", "C", median(rate, "c"), median(avg, "c")
  }' "$out/results"
