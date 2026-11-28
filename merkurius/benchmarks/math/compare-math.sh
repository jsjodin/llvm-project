#!/usr/bin/env bash
# Compares the math library's functions in a GPU loop (README.md):
# math_loop.m under merkc, whose device math is LLVM libc's libm, with the
# same loops in C with OpenMP offload, in HIP and in Fortran under Flang,
# whose device math is ROCm's ocml. All at -O2 on the same GPU, taking turns
# so that none gets the quieter part of the machine's time.
#
#   compare-math.sh [build-dir] [offload-arch] [rounds]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is this machine's GPU by default, as amdgpu-arch finds it;
# rounds, each a run of all four, 3 by default. ROCM_PATH is /opt/rocm by
# default. MATH_N (elements, 2^20), MATH_K (applications an element, 1000)
# and MATH_REPEAT (timed runs a function, the best kept, 5) pass through.
# MATH_PRECISION=32 compares the float32 versions, math_loop32.*, whose
# Merkurius program uses math_float32; 64, the default, math_loop.*.
# The CU mask is LIBOMPTARGET_AMDGPU_CU_MASK's, else half of the GPU's if
# it drives a display, as pipeline/sweep.sh's, and HIP's HSA_CU_MASK the
# same. On a machine that others share, run it through
# utils/gpu/with-gpus.py.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-$("$build/bin/amdgpu-arch" 2>/dev/null | head -n1)}"
[[ -n "$arch" ]] || { echo "$0: found no GPU; give its offload-arch" >&2; exit 1; }
rounds="${3:-3}"
rocm="${ROCM_PATH:-/opt/rocm}"
case "${MATH_PRECISION:-64}" in
  64) base=math_loop ;;
  32) base=math_loop32 ;;
  *) echo "$0: MATH_PRECISION is 32 or 64" >&2; exit 1 ;;
esac
out="${MATH_OUT:-$(mktemp -d)}"
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_TARGET_OFFLOAD=MANDATORY
mask="${LIBOMPTARGET_AMDGPU_CU_MASK-$(python3 -I "$here/../../utils/lit/gpu_info.py" unit-mask)}"
if [[ -n "$mask" ]]; then
  export LIBOMPTARGET_AMDGPU_CU_MASK="$mask"
  export HSA_CU_MASK="${HSA_CU_MASK:-0:$mask}"
fi

"$build/bin/merkc" -O2 -fopenmp -fdo-loops --offload-arch="$arch" \
  "${flags[@]}" "$here/$base.m" -o "$out/merkurius"
"$build/bin/clang" -O2 -fopenmp --offload-arch="$arch" "${flags[@]}" \
  "$here/$base.c" -lm -o "$out/c"
"$build/bin/clang" -x hip -O2 --offload-arch="$arch" --rocm-path="$rocm" \
  "$here/$base.hip" -L"$rocm/lib" -lamdhip64 -lm \
  -Wl,-rpath,"$rocm/lib" -o "$out/hip"
"$build/bin/flang" -O2 -fopenmp --offload-arch="$arch" "${flags[@]}" \
  "$here/$base.f90" -o "$out/fortran"

progs=(merkurius c hip fortran)
: > "$out/results"
for ((r = 1; r <= rounds; r++)); do
  for prog in "${progs[@]}"; do
    "$out/$prog" > "$out/$prog.$r"
    awk -v prog="$prog" '{ print prog, $1, $2, $3 }' "$out/$prog.$r" >> "$out/results"
  done
  echo "round $r of $rounds done" >&2
done

# Each program's best time a function over the rounds, its time against
# HIP's, and its sum's relative difference from HIP's: the libraries round
# differently, and the reductions add in different orders.
awk '
  {
    key = $1 SUBSEP $2
    if (!(key in best) || $3 < best[key]) best[key] = $3
    sum[key] = $4
    if (!($2 in seen)) { seen[$2] = 1; order[++nf] = $2 }
  }
  function abs(x) { return x < 0 ? -x : x }
  END {
    split("merkurius c hip fortran", ps, " ")
    split("Merkurius (libc);C OpenMP (ocml);HIP (ocml);Flang (ocml)", names, ";")
    printf "%-6s", "func"
    for (i = 1; i <= 4; i++) printf " %22s", names[i]
    printf "\n"
    for (j = 1; j <= nf; j++) {
      f = order[j]; h = best["hip", f]
      printf "%-6s", f
      for (i = 1; i <= 4; i++) {
        t = best[ps[i], f]
        printf " %10.6f s %8.2fx", t, t / h
      }
      printf "\n"
    }
    printf "\nRelative difference of the sums from HIP'"'"'s:\n"
    for (j = 1; j <= nf; j++) {
      f = order[j]; h = sum["hip", f]
      printf "%-6s", f
      for (i = 1; i <= 4; i++)
        printf " %22.2e", abs(sum[ps[i], f] - h) / abs(h)
      printf "\n"
    }
  }' "$out/results"
echo "Outputs: $out" >&2
