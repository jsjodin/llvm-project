#!/usr/bin/env bash
# Compares bandwidth.m under merkc with bandwidth.c under the same build's
# clang, both at -O2 on the same GPU, taking turns so that neither gets the
# quieter part of the machine's time.
#
#   compare-c.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is gfx1030 by default; rounds, each a run of both, 5 by
# default. BANDWIDTH_ARRAY_SIZE and BANDWIDTH_NUM_TIMES pass through.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-gfx1030}"
rounds="${3:-5}"
out="${BANDWIDTH_OUT:-$(mktemp -d)}"
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_TARGET_OFFLOAD=MANDATORY

"$build/bin/merkc" -O2 -fopenmp -fdo-loops --offload-arch="$arch" "${flags[@]}" \
  "$here/bandwidth.m" -o "$out/mercury"
"$build/bin/clang" -O2 -fopenmp --offload-arch="$arch" "${flags[@]}" \
  "$here/bandwidth.c" -lm -o "$out/c"

: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for lang in mercury c; do
    "$out/$lang" | awk -v lang="$lang" -v r="$r" \
      '$1 ~ /^(Copy|Mul|Add|Triad|Dot)$/ { print lang, r, $1, $2 }' >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# Each kernel's median bandwidth over the rounds, each round's being from its
# fastest run, as in the program's own table.
awk '
  { v[$1, $3, ++m[$1, $3]] = $4 }
  function median(lang, k,    n, i, j, t, s) {
    n = m[lang, k]
    for (i = 1; i <= n; i++) s[i] = v[lang, k, i]
    for (i = 2; i <= n; i++) {
      t = s[i]
      for (j = i - 1; j >= 1 && s[j] > t; j--) s[j + 1] = s[j]
      s[j + 1] = t
    }
    return n % 2 ? s[(n + 1) / 2] : (s[n / 2] + s[n / 2 + 1]) / 2
  }
  function range(lang, k,    n, i, lo, hi) {
    n = m[lang, k]; lo = hi = v[lang, k, 1]
    for (i = 2; i <= n; i++) {
      if (v[lang, k, i] < lo) lo = v[lang, k, i]
      if (v[lang, k, i] > hi) hi = v[lang, k, i]
    }
    return sprintf("%.0f-%.0f", lo, hi)
  }
  END {
    printf "%-7s %16s %16s\n", "Kernel", "Mercury", "C"
    split("Copy Mul Add Triad Dot", ks, " ")
    for (i = 1; i <= 5; i++) {
      k = ks[i]
      printf "%-7s %16.0f %16.0f\n", k, median("mercury", k), median("c", k)
      printf "%-7s %16s %16s\n", "", range("mercury", k), range("c", k)
    }
  }' "$out/results"
