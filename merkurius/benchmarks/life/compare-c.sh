#!/usr/bin/env bash
# Compares life.m under merkc with life.c under the same build's clang, with
# cells of 8 bytes, as Mercury's ints are, and of 1, as C would usually have
# them, all at -O2 on the same GPU, taking turns so that none gets the
# quieter part of the machine's time. Each must give the same cells.
#
#   compare-c.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is gfx1030 by default; rounds, each a run of all three, 5 by
# default. LIFE_SIZE and LIFE_STEPS pass through. On a machine that others
# share, run it through utils/gpu/with-gpus.py.
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

"$build/bin/merkc" -O2 -fopenmp -fdo-loops --offload-arch="$arch" "${flags[@]}" \
  "$here/life.m" -o "$out/mercury"
"$build/bin/clang" -O2 -fopenmp --offload-arch="$arch" "${flags[@]}" \
  "$here/life.c" -o "$out/c64"
"$build/bin/clang" -O2 -fopenmp --offload-arch="$arch" -DCELL=uint8_t \
  "${flags[@]}" "$here/life.c" -o "$out/c8"

: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for prog in mercury c64 c8; do
    "$out/$prog" > "$out/$prog.$r"
    awk -v prog="$prog" -v r="$r" '
      /^Population:/ { pop = $2 }
      /^Checksum:/ { sum = $2 }
      NF == 4 && $1 ~ /^[0-9.]+$/ { print prog, r, $1, $2, $4, pop, sum }
    ' "$out/$prog.$r" >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# The cells must agree; then each program's median over the rounds of its
# fastest step and its average step.
awk '
  { rate[$1, ++n[$1]] = $3; avg[$1, n[$1]] = $5; cells[$1 " " $6 " " $7] = 1 }
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
    split("", seen); m = 0
    for (c in cells) { split(c, f, " "); if (!(f[2] " " f[3] in seen)) { seen[f[2] " " f[3]] = 1; m++ } }
    if (m != 1) { print "the programs gave different cells"; exit 1 }
    printf "%-22s %12s %16s\n", "Program", "Gcells/sec", "Average (sec)"
    split("mercury c64 c8", ps, " ")
    split("Mercury (8-byte cells),C (int64_t),C (uint8_t)", names, ",")
    for (i = 1; i <= 3; i++)
      printf "%-22s %12.3f %16.6f\n", names[i], median(rate, ps[i]), median(avg, ps[i])
  }' "$out/results"
