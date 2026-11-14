#!/usr/bin/env bash
# Compares swe-sparse.m, the shallow water equations on sparse tiles in a
# GPU process, under merkc with swe-sparse.c, the same tiles in C with
# OpenMP offload, and swe-sparse.hip, in HIP, both with a kernel a step and
# the bookkeeping on the host, all at -O2 on the same GPU, taking turns so
# that none gets the quieter part of the machine's time. Each must give the
# same bits.
#
#   compare-swe.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is gfx1030 by default; rounds, each a run of all three, 5 by
# default. ROCM_PATH is /opt/rocm by default. SWE_WORLD, SWE_TILE,
# SWE_STEPS, SWE_CASE and SWE_CAPACITY pass through. The first round's runs
# also write each step's active tiles and time to $out/<program>.trace, for
# a chart. On a machine that others share, run it through
# utils/gpu/with-gpus.py.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-gfx1030}"
rounds="${3:-5}"
rocm="${ROCM_PATH:-/opt/rocm}"
out="${SWE_OUT:-$(mktemp -d)}"
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_TARGET_OFFLOAD=MANDATORY

"$build/bin/merkc" -O2 -fopenmp -fdo-loops -fprocess-declarations \
  --offload-arch="$arch" "${flags[@]}" "$here/swe-sparse.m" -o "$out/mercury"
"$build/bin/clang" -O2 -ffp-contract=off -fopenmp --offload-arch="$arch" \
  "${flags[@]}" "$here/swe-sparse.c" -lm -o "$out/c"
"$build/bin/clang" -x hip -O2 -ffp-contract=off --offload-arch="$arch" \
  --rocm-path="$rocm" "$here/swe-sparse.hip" -L"$rocm/lib" -lamdhip64 -lm \
  -Wl,-rpath,"$rocm/lib" -o "$out/hip"

: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for prog in mercury c hip; do
    if ((r == 1)); then
      SWE_TRACE=1 "$out/$prog" > "$out/$prog.$r"
      grep '^step ' "$out/$prog.$r" > "$out/$prog.trace"
    else
      "$out/$prog" > "$out/$prog.$r"
    fi
    # Min, max, average and bookkeeping, after what must agree.
    awk -v prog="$prog" -v r="$r" '
      /^Time:/ { time = $2 }
      /^Wet cells:/ { wet = $3 }
      /^Checksum:/ { sum = $2 }
      /^Active tiles:/ { peak = $3; last = $6 }
      NF == 4 && $1 ~ /^[0-9.]+$/ {
        print prog, r, $3, $4, time "/" wet "/" sum "/" peak "/" last
      }
    ' "$out/$prog.$r" >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# The outcomes must agree; then each program's median over the rounds of
# its average step and of the bookkeeping's part of it.
awk '
  { avg[$1, ++n[$1]] = $3; keep[$1, n[$1]] = $4; outcome[$5] = 1 }
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
    for (o in outcome) { m++; last = o }
    if (m != 1) { print "the programs gave different results"; exit 1 }
    split(last, f, "/")
    printf "Time %s s, %s wet cells, checksum %s, tiles %s at most, %s at the end\n", f[1], f[2], f[3], f[4], f[5]
    printf "%-30s %14s %16s\n", "Program", "Average (sec)", "Bookkeeping"
    split("mercury c hip", ps, " ")
    split("Mercury (GPU process);C (OpenMP offload);HIP", names, ";")
    for (i = 1; i <= 3; i++)
      printf "%-30s %14.6f %16.6f\n", names[i], median(avg, ps[i]), median(keep, ps[i])
  }' "$out/results"
echo "Traces: $out/mercury.trace, $out/c.trace, $out/hip.trace" >&2
