#!/usr/bin/env bash
# Builds the benchmark of arrays streamed between GPU processes on two GPUs,
# with no host on the way, and runs it once (README.md). Run it with two
# GPUs visible:
#
#   utils/gpu/with-gpus.py --gpus A,B -- run.sh [build-dir] [offload-arch]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is the GPU's, this machine's by default, as amdgpu-arch finds
# it. GPU_LINK_MIB sets the mebibytes that each measurement moves, 256 by
# default. GPU_LINK_WORKERS=K gives each source and sink worker teams, up to
# a K'th of its GPU's kernel's teams (link.m, wsource and wsink).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-$("$build/bin/amdgpu-arch" 2>/dev/null | head -n1)}"
[[ -n "$arch" ]] || { echo "$0: found no GPU; give its offload-arch" >&2; exit 1; }
mib="${GPU_LINK_MIB:-256}"
out="${GPU_LINK_OUT:-$(mktemp -d)}"
# libomp is built in the runtimes' tree, not installed into lib.
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Kinds with worker teams need OpenMP's loops, and so -fopenmp only then.
omp_flags=(-fdo-loops)
[[ -n "${GPU_LINK_WORKERS:-}" ]] && omp_flags+=(-fopenmp)
"$build/bin/merkc" -O2 "${omp_flags[@]}" -fprocess-declarations --offload-arch="$arch" \
  "${flags[@]}" "$here/link.m" -o "$out/link"
"$out/link" "$mib" ${GPU_LINK_WORKERS:+workers "$GPU_LINK_WORKERS"}
