#!/usr/bin/env bash
# Builds the open-holes benchmark and runs it five times (README.md).
#
#   run.sh [build-dir] [N] [R]
#
# build-dir is the Merkurius build, ../../../../build-merk by default; the
# list has N elements, 100000 by default, built and summed R times, 100.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
n="${2:-100000}"
r="${3:-100}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$build/bin/merkc" -O2 -fopen-types "$here/open-holes.m" -o "$out/open-holes"
# Temporary (open types step 5, docs/INPROGRESS.md): the collector cannot
# yet move open terms nor the trail's entries, which every binding here makes
# (the condition in `rounds` is live), so one chunk holds the whole run and
# no collection is ever due. It is about 30 bytes an element.
export MERK_HEAP_CHUNK_BYTES=$((n * r * 40 + 16777216))
for _ in 1 2 3 4 5; do
  /usr/bin/time -f "%e s elapsed, %U s user, %M KB peak" "$out/open-holes" "$n" "$r"
done
