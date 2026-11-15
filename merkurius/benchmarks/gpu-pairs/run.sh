#!/usr/bin/env bash
# Builds the benchmark of arrays between GPU processes on two GPUs and runs
# it once (README.md). Run it with two GPUs visible:
#
#   utils/gpu/with-gpus.py --gpus A,B -- run.sh [build-dir] [offload-arch]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is the GPU's, this machine's by default, as amdgpu-arch finds
# it. GPU_PAIRS_MIB sets the mebibytes that each measurement moves, 64 by
# default. GPU_PAIRS_WORKERS=K gives each relay worker teams, up to a K'th
# of its GPU's kernel's teams (gpupairs.m, wrelay).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-$("$build/bin/amdgpu-arch" 2>/dev/null | head -n1)}"
[[ -n "$arch" ]] || { echo "$0: found no GPU; give its offload-arch" >&2; exit 1; }
mib="${GPU_PAIRS_MIB:-64}"
out="${GPU_PAIRS_OUT:-$(mktemp -d)}"
# libomp is built in the runtimes' tree, not installed into lib.
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Relays with worker teams need OpenMP's loops, and so -fopenmp only then.
omp_flags=(-fdo-loops)
[[ -n "${GPU_PAIRS_WORKERS:-}" ]] && omp_flags+=(-fopenmp)
"$build/bin/merkc" -O2 "${omp_flags[@]}" -fprocess-declarations --offload-arch="$arch" \
  "${flags[@]}" "$here/gpupairs.m" -o "$out/gpupairs"
"$out/gpupairs" "$mib" ${GPU_PAIRS_WORKERS:+workers "$GPU_PAIRS_WORKERS"}
