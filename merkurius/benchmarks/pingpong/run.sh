#!/usr/bin/env bash
# Builds the ping-pong latency benchmark and runs its measurements: node
# 1's main and node 1's target region, each against node 2 over TCP on
# loopback and against an echo thread in node 1; and the floors under them,
# plain C round trips over TCP, through condition variables, and over a
# GPU's RPC server (README.md).
#
#   run.sh [build-dir] [offload-arch]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is the GPU's, such as gfx1030, and without it the GPU's
# measurements are left out. PINGPONG_ROUNDS sets the rounds, 10000 by
# default.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-}"
rounds="${PINGPONG_ROUNDS:-10000}"
out="${PINGPONG_OUT:-$(mktemp -d)}"
# libomp is built in the runtimes' tree, not installed into lib.
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
llvm="$(cd "$here/../../.." && pwd)"

"$build/bin/clang" -O2 "$here/tcp-floor.c" -o "$out/tcp-floor"
"$build/bin/clang" -O2 "$here/condvar-floor.c" -lpthread -o "$out/condvar-floor"

if [[ -n "$arch" ]]; then
  "$build/bin/merkc" -O2 -fopenmp --offload-arch="$arch" "${flags[@]}" \
    "$here/pingpong.m" -o "$out/pingpong"
  "$build/bin/clang++" -O2 -std=c++17 -fopenmp --offload-arch="$arch" \
    -I"$llvm/libc" "${flags[@]}" "$here/rpc-floor.cpp" -o "$out/rpc-floor"
  export OMP_TARGET_OFFLOAD=MANDATORY
  python3 -I "$here/pair.py" "$out/pingpong" cpu "$rounds"
  python3 -I "$here/pair.py" "$out/pingpong" gpu "$rounds"
  "$out/pingpong" cpu-local "$rounds"
  "$out/pingpong" gpu-local "$rounds"
  "$out/tcp-floor" "$rounds"
  "$out/condvar-floor" "$rounds"
  "$out/rpc-floor" "$rounds"
else
  "$build/bin/merkc" -O2 "$here/pingpong.m" -o "$out/pingpong"
  python3 -I "$here/pair.py" "$out/pingpong" cpu "$rounds"
  "$out/pingpong" cpu-local "$rounds"
  "$out/tcp-floor" "$rounds"
  "$out/condvar-floor" "$rounds"
  echo "the GPU's measurements: skipped, no offload-arch given"
fi
