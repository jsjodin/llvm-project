#!/usr/bin/env bash
# Compares life-places.m, Life on sparse tiles on several places of one GPU,
# each a GPU process, with life-sparse.m, the same tiles in one GPU process,
# both under merkc at -O2 with -fopenmp, taking turns so that none gets the
# quieter part of the machine's time. Each must give the same cells.
#
#   compare-places.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is this machine's GPU by default, as amdgpu-arch finds it;
# rounds, each a run of every program, 5 by default. LIFE_PLACES_LIST is the
# numbers of places to run, "1 2 4" by default, and LIFE_MOVES_LIST the most
# tiles moved a step, "4" by default, each with each. LIFE_WORLD (8192),
# LIFE_TILE (64), LIFE_SOUP (256), LIFE_STEPS (1000) and LIFE_CAPACITY pass
# through. On a machine that others share, run it through
# utils/gpu/with-gpus.py.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-$("$build/bin/amdgpu-arch" 2>/dev/null | head -n1)}"
[[ -n "$arch" ]] || { echo "$0: found no GPU; give its offload-arch" >&2; exit 1; }
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
read -r -a places <<< "${LIFE_PLACES_LIST:-1 2 4}"
read -r -a moves <<< "${LIFE_MOVES_LIST:-4}"

"$build/bin/merkc" -O2 -fopenmp -fdo-loops -fprocess-declarations \
  --offload-arch="$arch" "${flags[@]}" "$here/life-sparse.m" -o "$out/sparse"
"$build/bin/merkc" -O2 -fopenmp -fdo-loops -fprocess-declarations \
  --offload-arch="$arch" "${flags[@]}" "$here/life-places.m" -o "$out/places"

# The programs, one a line: a name, then its settings.
programs=("sparse")
for p in "${places[@]}"; do
  for m in "${moves[@]}"; do
    programs+=("places-$p-$m LIFE_PLACES=$p LIFE_MOVES=$m")
  done
done

: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for program in "${programs[@]}"; do
    read -r -a words <<< "$program"
    name="${words[0]}"
    binary="${name%%-*}"
    env "${words[@]:1}" "$out/$binary" > "$out/$name.$r"
    # Both print min, max and average, and life-sparse.m its bookkeeping.
    awk -v name="$name" -v r="$r" '
      /^Population:/ { pop = $2 }
      /^Checksum:/ { sum = $2 }
      /^Tiles moved:/ { moved = $3 }
      NF >= 3 && $1 ~ /^[0-9.]+$/ { print name, r, $3, pop, sum, moved + 0 }
    ' "$out/$name.$r" >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# The cells must agree; then each program's median over the rounds of its
# average step.
awk '
  { if (!($1 in n)) order[++k] = $1
    avg[$1, ++n[$1]] = $3; cells[$4 " " $5] = 1; moved[$1] = $6 }
  function median(p,    c, i, j, t, s) {
    c = n[p]
    for (i = 1; i <= c; i++) s[i] = avg[p, i]
    for (i = 2; i <= c; i++) {
      t = s[i]
      for (j = i - 1; j >= 1 && s[j] > t; j--) s[j + 1] = s[j]
      s[j + 1] = t
    }
    return c % 2 ? s[(c + 1) / 2] : (s[c / 2] + s[c / 2 + 1]) / 2
  }
  END {
    m = 0
    for (c in cells) m++
    if (m != 1) { print "the programs gave different cells"; exit 1 }
    printf "%-26s %14s %12s\n", "Program", "Average (sec)", "Tiles moved"
    for (i = 1; i <= k; i++)
      printf "%-26s %14.6f %12s\n", order[i], median(order[i]), moved[order[i]]
  }' "$out/results"
echo "Outputs: $out" >&2
