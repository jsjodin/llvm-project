#!/usr/bin/env bash
# Builds the open-unify benchmark and runs it five times (README.md).
#
#   run.sh [build-dir] [N] [R]
#
# build-dir is the Merkurius build, ../../../../build-merk by default; the
# lists have N elements, 100000 by default, and are built, unified and summed
# R times, 100.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
n="${2:-100000}"
r="${3:-100}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$build/bin/merkc" -O2 -fopen-types "$here/open-unify.m" -o "$out/open-unify"
for _ in 1 2 3 4 5; do
  /usr/bin/time -f "%e s elapsed, %U s user, %M KB peak" "$out/open-unify" "$n" "$r"
done
