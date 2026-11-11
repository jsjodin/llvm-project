#!/usr/bin/env bash
# Builds Game of Life three ways, sequential, OpenMP on the host's threads
# and OpenMP on a GPU, and runs each.
#
#   run.sh [build-dir] [offload-arch]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is the GPU's, such as gfx1030, and without it the GPU build
# is left out. LIFE_SIZE and LIFE_STEPS pass through to the program;
# OMP_NUM_THREADS sets the host's threads. On a machine that others share,
# run it through utils/gpu/with-gpus.py.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-}"
out="${LIFE_OUT:-$(mktemp -d)}"
merkc="$build/bin/merkc"
# libomp is built in the runtimes' tree, not installed into lib.
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
src="$here/life.m"

echo "== sequential"
"$merkc" -O2 -fdo-loops "$src" -o "$out/sequential"
"$out/sequential"

echo "== OpenMP, host"
"$merkc" -O2 -fopenmp -fdo-loops "${flags[@]}" "$src" -o "$out/host"
OMP_TARGET_OFFLOAD=DISABLED "$out/host"

if [[ -n "$arch" ]]; then
  echo "== OpenMP, GPU ($arch)"
  "$merkc" -O2 -fopenmp -fdo-loops --offload-arch="$arch" "${flags[@]}" "$src" -o "$out/gpu"
  OMP_TARGET_OFFLOAD=MANDATORY "$out/gpu"
else
  echo "== OpenMP, GPU: skipped, no offload-arch given"
fi
