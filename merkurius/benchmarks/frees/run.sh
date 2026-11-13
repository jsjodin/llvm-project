#!/usr/bin/env bash
# Builds the benchmark of a GPU process's blocks and runs each measurement
# three times (README.md).
#
#   run.sh [build-dir] [offload-arch]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is the GPU's, gfx1030 by default.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-gfx1030}"
out="$(mktemp -d)"
# libomp is built in the runtimes' tree, not installed into lib.
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$build/bin/merkc" -O2 -fprocess-declarations --offload-arch="$arch" \
  "${flags[@]}" "$here/frees.m" -o "$out/frees"
for m in "ping 10000" "none 100000" "alloc 100000 4096" "malloc 100000 4096" \
    "alloc 10000 4194304" "malloc 10000 4194304" "cold-alloc 16 4194304" \
    "cold-malloc 16 4194304" "array 100000 512" "array 100 524288"; do
  for _ in 1 2 3; do
    # shellcheck disable=SC2086
    "$out/frees" gpu $m
  done
done
