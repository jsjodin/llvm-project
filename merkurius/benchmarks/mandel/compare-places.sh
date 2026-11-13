#!/usr/bin/env bash
# Compares mandel-places.m's places under each policy with the dense
# programs, mandel.m in one GPU process and mandel.c a kernel a frame, all at
# -O2 on the same GPU, taking turns so that none gets the quieter part of the
# machine's time. They must give the same iterations.
#
#   compare-places.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is gfx1030 by default; rounds, each a run of every program,
# 5 by default. MANDEL_SIZE, MANDEL_FRAMES, MANDEL_TILE, MANDEL_PLACES and
# MANDEL_CHUNK pass through. On a machine that others share, run it through
# utils/gpu/with-gpus.py.
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

mercury=("$build/bin/merkc" -O2 -fopenmp -fdo-loops -fprocess-declarations
  --offload-arch="$arch" "${flags[@]}")
"${mercury[@]}" "$here/mandel.m" -o "$out/dense"
"${mercury[@]}" "$here/mandel-places.m" -o "$out/places"
"$build/bin/clang" -O2 -ffp-contract=off -fopenmp --offload-arch="$arch" \
  "${flags[@]}" "$here/mandel.c" -o "$out/c"

programs=(dense c static steal pool)
: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for prog in "${programs[@]}"; do
    case "$prog" in
      dense | c) "$out/$prog" > "$out/$prog.$r" ;;
      *) MANDEL_POLICY="$prog" "$out/places" > "$out/$prog.$r" ;;
    esac
    awk -v prog="$prog" '
      /^Iterations:/ { total = $2 }
      /^Checksum:/ { sum = $2 }
      /^Tiles handed:/ { handed = $3 }
      /^Balance:/ { balance = $2 }
      NF == 2 && $1 ~ /^[0-9.]+$/ {
        print prog, $1, $2, total, sum, (balance == "" ? "-" : balance),
          (handed == "" ? "-" : handed)
      }
    ' "$out/$prog.$r" >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# The iterations must agree; then each program's median time, rate, balance
# (the busiest place's time over the average) and tiles handed over.
awk '
  {
    k = ++n[$1]
    time[$1, k] = $2; rate[$1, k] = $3; balance[$1, k] = $6; handed[$1, k] = $7
    cells[$4 " " $5] = 1
  }
  function median(a, p,    k, i, j, t, s) {
    k = n[p]
    if (a[p, 1] == "-") return "-"
    for (i = 1; i <= k; i++) s[i] = a[p, i]
    for (i = 2; i <= k; i++) {
      t = s[i]
      for (j = i - 1; j >= 1 && s[j] + 0 > t + 0; j--) s[j + 1] = s[j]
      s[j + 1] = t
    }
    return k % 2 ? s[(k + 1) / 2] : (s[k / 2] + s[k / 2 + 1]) / 2
  }
  END {
    m = 0
    for (c in cells) m++
    if (m != 1) { print "the programs gave different iterations"; exit 1 }
    printf "%-28s %10s %14s %8s %8s\n", "Program", "Time (sec)", "Giterations/s",
      "Balance", "Handed"
    split("dense c static steal pool", ps, " ")
    split("Mercury, one GPU process;C, a kernel a frame;Places, static;" \
      "Places, steal;Places, pool", names, ";")
    for (i = 1; i <= 5; i++)
      printf "%-28s %10.4f %14.3f %8s %8s\n", names[i], median(time, ps[i]),
        median(rate, ps[i]), median(balance, ps[i]), median(handed, ps[i])
  }' "$out/results"
