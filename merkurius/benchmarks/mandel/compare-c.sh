#!/usr/bin/env bash
# Compares mandel.m, the zoom in one GPU process, under merkc with mandel.c,
# a kernel a frame, under the same build's clang, both at -O2 on the same
# GPU, taking turns so that neither gets the quieter part of the machine's
# time. They must give the same iterations.
#
#   compare-c.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is gfx1030 by default; rounds, each a run of both, 5 by
# default. MANDEL_SIZE and MANDEL_FRAMES pass through. On a machine that
# others share, run it through utils/gpu/with-gpus.py.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-gfx1030}"
rounds="${3:-5}"
out="${MANDEL_OUT:-$(mktemp -d)}"
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_TARGET_OFFLOAD=MANDATORY

"$build/bin/merkc" -O2 -fopenmp -fdo-loops -fprocess-declarations \
  --offload-arch="$arch" "${flags[@]}" "$here/mandel.m" -o "$out/mercury"
"$build/bin/clang" -O2 -ffp-contract=off -fopenmp --offload-arch="$arch" \
  "${flags[@]}" "$here/mandel.c" -o "$out/c"

: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for prog in mercury c; do
    "$out/$prog" > "$out/$prog.$r"
    awk -v prog="$prog" '
      /^Iterations:/ { total = $2 }
      /^Checksum:/ { sum = $2 }
      NF == 2 && $1 ~ /^[0-9.]+$/ { print prog, $1, $2, total, sum }
    ' "$out/$prog.$r" >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# The iterations must agree; then each program's median time over the
# rounds, and its rate.
awk '
  { time[$1, ++n[$1]] = $2; rate[$1, n[$1]] = $3; cells[$4 " " $5] = 1 }
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
    if (m != 1) { print "the programs gave different iterations"; exit 1 }
    printf "%-26s %12s %16s\n", "Program", "Time (sec)", "Giterations/s"
    split("mercury c", ps, " ")
    split("Mercury (GPU process);C (kernel a frame)", names, ";")
    for (i = 1; i <= 2; i++)
      printf "%-26s %12.4f %16.3f\n", names[i], median(time, ps[i]), median(rate, ps[i])
  }' "$out/results"
