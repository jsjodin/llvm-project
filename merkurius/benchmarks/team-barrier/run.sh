#!/usr/bin/env bash
# Builds the benchmark of a barrier between teams and runs its
# measurements: the three shapes, over numbers of teams, on the GPU and on the
# host's teams, which libomp runs as threads (README.md).
#
#   run.sh [build-dir] [offload-arch]
#
# build-dir is the Merkurius build, ../../../../build-merk by default;
# offload-arch is the GPU's, such as gfx1030, and without it the GPU's
# measurements are left out. TEAM_BARRIER_ROUNDS sets the rounds, 10000 by
# default, and TEAM_BARRIER_TEAMS the numbers of teams, "2 4 16 64" by
# default.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
arch="${2:-}"
rounds="${TEAM_BARRIER_ROUNDS:-10000}"
teams="${TEAM_BARRIER_TEAMS:-2 4 16 64}"
out="${TEAM_BARRIER_OUT:-$(mktemp -d)}"
# libomp is built in the runtimes' tree, not installed into lib.
omp="$build/runtimes/runtimes-bins/openmp/runtime/src"
libs=("$omp" "$build/lib" "$build/lib/x86_64-unknown-linux-gnu")
flags=()
for d in "${libs[@]}"; do flags+=("-L$d"); done
export LD_LIBRARY_PATH="$(IFS=:; echo "${libs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -n "$arch" ]]; then
  "$build/bin/merkc" -O2 -fopenmp --offload-arch="$arch" "${flags[@]}" \
    "$here/teambarrier.m" -o "$out/teambarrier"
  echo "on the GPU"
  for shape in central tree library; do
    for n in $teams; do
      OMP_TARGET_OFFLOAD=MANDATORY "$out/teambarrier" "$shape" "$n" "$rounds"
    done
  done
else
  "$build/bin/merkc" -O2 -fopenmp "${flags[@]}" "$here/teambarrier.m" \
    -o "$out/teambarrier"
  echo "the GPU's measurements: skipped, no offload-arch given"
fi
echo "on the host's teams"
for shape in central tree library; do
  for n in $teams; do
    OMP_TARGET_OFFLOAD=DISABLED "$out/teambarrier" "$shape" "$n" "$rounds"
  done
done
