#!/usr/bin/env bash
# Builds the benchmark of arrays as messages and runs its measurements:
# node 1's main and node 1's target region, against node 2 over TCP on
# loopback and against node 1's main, and the floor under the GPU's,
# omp_target_memcpy alone (README.md).
#
#   run.sh [build-dir] [offload-arch]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is the GPU's, such as gfx1030, and without it the GPU's
# measurements are left out. ARRAY_MESSAGES_MIB sets the mebibytes that each
# measurement moves, 64 by default.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-}"
mib="${ARRAY_MESSAGES_MIB:-64}"
out="${ARRAY_MESSAGES_OUT:-$(mktemp -d)}"
# libomp is built in the runtimes' tree, not installed into lib.
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pair="$here/../pingpong/pair.py"

if [[ -n "$arch" ]]; then
  "$build/bin/merkc" -O2 -fopenmp --offload-arch="$arch" "${flags[@]}" \
    "$here/arraymessages.m" -o "$out/arraymessages"
  "$build/bin/clang++" -O2 -std=c++17 -fopenmp --offload-arch="$arch" \
    "${flags[@]}" "$here/memcpy-floor.cpp" -o "$out/memcpy-floor"
  export OMP_TARGET_OFFLOAD=MANDATORY
  python3 -I "$pair" "$out/arraymessages" cpu "$mib"
  python3 -I "$pair" "$out/arraymessages" gpu "$mib"
  "$out/arraymessages" cpu-local "$mib"
  "$out/arraymessages" gpu-local "$mib"
  "$out/memcpy-floor" "$mib"
else
  "$build/bin/merkc" -O2 "$here/arraymessages.m" -o "$out/arraymessages"
  python3 -I "$pair" "$out/arraymessages" cpu "$mib"
  "$out/arraymessages" cpu-local "$mib"
  echo "the GPU's measurements: skipped, no offload-arch given"
fi
