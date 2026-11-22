#!/usr/bin/env bash
# Compares mandel-recover.m with mandel-places.m under the pool policy, its
# plain run, at -O2 on the same GPU, taking turns (docs/INPROGRESS.md,
# Failure, step 4, D1): first with no faults, the cost of keeping each
# place's chunk and monitoring the places; then with MERK_FAULTS failing
# places at random with the mean time between failures MTBF_MS, the cost of
# recovering. They must give the same iterations and checksum.
#
#   compare-recover.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is this machine's GPU by default, as amdgpu-arch finds it;
# rounds, each a run of every program, 5 by default. MTBF_MS is 100 by
# default, and each round's faults have the round's seed. MANDEL_SIZE,
# MANDEL_FRAMES, MANDEL_TILE, MANDEL_PLACES and MANDEL_CHUNK pass through.
# MANDEL_FAULTS=kernel ends whole kernels instead of places. On a machine
# that others share, run it through utils/gpu/with-gpus.py.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-$("$build/bin/amdgpu-arch" 2>/dev/null | head -n1)}"
[[ -n "$arch" ]] || { echo "$0: found no GPU; give its offload-arch" >&2; exit 1; }
rounds="${3:-5}"
mtbf="${MTBF_MS:-100}"
case "${MANDEL_FAULTS:-place}" in
  place) faults="fail=order" ;;
  kernel) faults="kernel" ;;
  *) echo "$0: MANDEL_FAULTS is place or kernel" >&2; exit 1 ;;
esac
out="${MANDEL_OUT:-$(mktemp -d)}"
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_TARGET_OFFLOAD=MANDATORY

merkurius=("$build/bin/merkc" -O2 -fopenmp -fdo-loops -fprocess-declarations
  --offload-arch="$arch" "${flags[@]}")
"${merkurius[@]}" "$here/mandel-places.m" -o "$out/places"
"${merkurius[@]}" "$here/mandel-recover.m" -o "$out/recover"

programs=(pool recover faults)
: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for prog in "${programs[@]}"; do
    case "$prog" in
      pool) MANDEL_POLICY=pool "$out/places" > "$out/$prog.$r" ;;
      recover) "$out/recover" > "$out/$prog.$r" ;;
      faults)
        MERK_FAULTS="seed=$r mtbf=$mtbf $faults count=1000000" \
          "$out/recover" > "$out/$prog.$r" 2> "$out/$prog.$r.faults" ;;
    esac
    awk -v prog="$prog" '
      /^Iterations:/ { total = $2 }
      /^Checksum:/ { sum = $2 }
      /^Places lost:/ { lost = $3 }
      /^Tiles given back:/ { back = $4 }
      NF == 2 && $1 ~ /^[0-9.]+$/ {
        print prog, $1, $2, total, sum, (lost == "" ? "-" : lost),
          (back == "" ? "-" : back)
      }
    ' "$out/$prog.$r" >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# The iterations and checksums must agree; then each program's median time,
# rate, places lost and tiles given back.
awk -v mtbf="$mtbf" '
  {
    k = ++n[$1]
    time[$1, k] = $2; rate[$1, k] = $3; lost[$1, k] = $6; back[$1, k] = $7
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
    printf "%-34s %10s %14s %6s %6s\n", "Program", "Time (sec)",
      "Giterations/s", "Lost", "Back"
    split("pool recover faults", ps, " ")
    names[1] = "Places, pool"
    names[2] = "Recovering, no faults"
    names[3] = "Recovering, MTBF " mtbf " ms"
    for (i = 1; i <= 3; i++)
      printf "%-34s %10.4f %14.3f %6s %6s\n", names[i], median(time, ps[i]),
        median(rate, ps[i]), median(lost, ps[i]), median(back, ps[i])
  }' "$out/results"
