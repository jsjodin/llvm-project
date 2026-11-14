#!/usr/bin/env bash
# Times pipeline.m, as three GPU processes and as one, and pipeline.hip, over
# frame sizes, rounds of work a stage and feedback, all at -O2 on the same
# GPU, taking turns so that none gets the quieter part of the machine's
# time. Each configuration must give one checksum. Prints each one's median
# time a frame in microseconds.
#
#   sweep.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is gfx1030 by default; rounds 3 by default. ROCM_PATH is
# /opt/rocm by default. PIPE_SIZES, PIPE_WORKS and PIPE_FEEDBACKS are the
# lists to sweep ("4096 65536 1048576", "16 256 4096", "0 1"); PIPE_FRAMES
# (40) passes through. On a machine that others share, run it through
# utils/gpu/with-gpus.py.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-gfx1030}"
rounds="${3:-3}"
rocm="${ROCM_PATH:-/opt/rocm}"
out="${PIPE_OUT:-$(mktemp -d)}"
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PIPE_FRAMES="${PIPE_FRAMES:-40}"
# The GPU processes' and HIP's queues on the same compute units.
export LIBOMPTARGET_AMDGPU_CU_MASK="${LIBOMPTARGET_AMDGPU_CU_MASK:-0-39}"
export HSA_CU_MASK="${HSA_CU_MASK:-0:$LIBOMPTARGET_AMDGPU_CU_MASK}"

"$build/bin/merkc" -O2 -fopenmp -fdo-loops -fprocess-declarations \
  --offload-arch="$arch" "${flags[@]}" "$here/pipeline.m" -o "$out/mercury"
"$build/bin/clang" -x hip -O2 -ffp-contract=off --offload-arch="$arch" \
  --rocm-path="$rocm" "$here/pipeline.hip" -L"$rocm/lib" -lamdhip64 \
  -Wl,-rpath,"$rocm/lib" -o "$out/hip"

: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for words in ${PIPE_SIZES:-4096 65536 1048576}; do
    for work in ${PIPE_WORKS:-16 256 4096}; do
      for feedback in ${PIPE_FEEDBACKS:-0 1}; do
        for mode in pipeline single hip; do
          prog="$out/mercury"
          [[ $mode == hip ]] && prog="$out/hip"
          PIPE_WORDS=$words PIPE_WORK=$work PIPE_FEEDBACK=$feedback \
            PIPE_MODE=$mode "$prog" > "$out/last"
          awk -v c="$words $work $feedback $mode" '
            /^Checksum:/ { sum = $2 }
            /^Time a frame:/ { t = $4 }
            END { print c, t, sum }' "$out/last" >> "$out/results"
        done
      done
    done
  done
  echo "round $r of $rounds done" >&2
done

awk '
  { key = $1 " " $2 " " $3; t[key, $4, ++n[key, $4]] = $5
    sums[key, $6] = 1; keys[key] = 1 }
  function median(key, m,    k, i, j, x, s) {
    k = n[key, m]
    for (i = 1; i <= k; i++) s[i] = t[key, m, i]
    for (i = 2; i <= k; i++) {
      x = s[i]
      for (j = i - 1; j >= 1 && s[j] > x; j--) s[j + 1] = s[j]
      s[j + 1] = x
    }
    return k % 2 ? s[(k + 1) / 2] : (s[k / 2] + s[k / 2 + 1]) / 2
  }
  END {
    for (key in keys) {
      m = 0
      for (ks in sums) { split(ks, p, SUBSEP); if (p[1] == key) m++ }
      if (m != 1) { print "different checksums for " key; exit 1 }
    }
    printf "%8s %6s %8s %12s %12s %12s\n", "Words", "Work", "Feedback",
      "Pipeline", "Single", "HIP"
    for (key in keys) {
      split(key, f, " ")
      printf "%8s %6s %8s %12.1f %12.1f %12.1f\n", f[1], f[2], f[3],
        median(key, "pipeline"), median(key, "single"), median(key, "hip")
    }
  }' "$out/results" | sort -k1,1n -k2,2n -k3,3n
