#!/usr/bin/env bash
# Coverage of Merkurius's code by a set of lit tests (docs/INPROGRESS.md, A
# conformance suite): runs them with the coverage build (build_merk.sh's
# help says how to configure it), whose merkc and host runtime are
# instrumented, merges the profiles and writes llvm-cov's reports.
#
#   coverage.sh -o OUT [-b BUILD] [-t TOOLS] [LIT-PATH...]
#
# OUT gets lit.log, merged.profdata, report.txt (by file), functions.txt
# (by function), summary.json (llvm-cov export -summary-only, to
# compare two runs) and html/ (lines and regions, from index.html). BUILD is
# the coverage build, ../build-cov beside the repository by default; TOOLS
# the build whose clang compiled it, which has llvm-profdata and llvm-cov,
# ../build-merk by default. LIT-PATH defaults to all of Merkurius's tests.
# GPU tests are left out, as on any host run; the device runtime is not
# measured.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$(cd "$here/../.." && pwd)"
work="$(cd "$src/../.." && pwd)"
build="$work/build-cov"
tools="$work/build-merk"
out=""
while getopts "o:b:t:" option; do
  case $option in
    o) out="$OPTARG" ;;
    b) build="$OPTARG" ;;
    t) tools="$OPTARG" ;;
    *) sed -n '2,17p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [[ -z "$out" ]]; then
  sed -n '2,17p' "$0"
  exit 2
fi
if ! grep -q '^MERKURIUS_COVERAGE:BOOL=ON' "$build/CMakeCache.txt"; then
  echo "coverage.sh: $build is not a build with MERKURIUS_COVERAGE=ON" >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  set -- "$build/tools/merkurius/test"
fi

mkdir -p "$out"
out="$(cd "$out" && pwd)"
rm -rf "$out/raw" "$out/html"
mkdir "$out/raw"

# A file for each process (%p): merging online (%m) is only for copies of
# one binary, and the test programs, which link the same runtime objects
# differently, collide in it, and then print a warning into their output.
# The raw files, a few GB, go once merged.
LLVM_PROFILE_FILE="$out/raw/%p.profraw" "$build/bin/llvm-lit" -sv \
  --filter-out='-gpu\.test$' "$@" > "$out/lit.log" 2>&1 || true
grep -E '^ *(Passed|Failed|Unsupported|Excluded) *:' "$out/lit.log" || true

find "$out/raw" -name '*.profraw' > "$out/profraw-files.txt"
# Tests that kill a program on purpose leave its profile incomplete; the
# merge skips those, and says how many it did.
"$tools/bin/llvm-profdata" merge -sparse -j 32 --failure-mode=all \
  -f "$out/profraw-files.txt" -o "$out/merged.profdata" 2> "$out/merge.log"
echo "coverage.sh: merged $(wc -l < "$out/profraw-files.txt") profiles," \
  "skipped $(grep -c 'invalid instrumentation profile' "$out/merge.log" || true)"
rm -rf "$out/raw"

# The instrumented code: merkc, Merkurius's shared libraries, and the host
# runtime's objects, which programs link statically.
objects=("$build/bin/merkc")
for library in "$build"/lib/libMerkurius*.so; do
  objects+=(-object "$library")
done
for object in "$build"/tools/merkurius/runtime/CMakeFiles/merkrt.dir/*.o; do
  objects+=(-object "$object")
done
sources=("$src/lib" "$src/include" "$src/runtime" "$src/tools")
# llvm-cov warns that some thousands of functions have mismatched data.
# Each is a translation unit's record, of hash 0, of an inline or template
# function it could see and did not use; the copy that a unit used is
# counted, so nothing is lost (checked 2026-11-23: every one has a counted
# copy, with llvm-cov report -dump).
cov=("$tools/bin/llvm-cov")
common=(-instr-profile="$out/merged.profdata" "${objects[@]}")

"${cov[@]}" report "${common[@]}" -sources "${sources[@]}" \
  > "$out/report.txt"
"${cov[@]}" report -show-functions "${common[@]}" -sources "${sources[@]}" \
  > "$out/functions.txt"
"${cov[@]}" export -summary-only "${common[@]}" -sources "${sources[@]}" \
  > "$out/summary.json"
"${cov[@]}" show -format=html -output-dir="$out/html" \
  -show-line-counts-or-regions -show-branches=count "${common[@]}" \
  -sources "${sources[@]}"
tail -1 "$out/report.txt"
echo "coverage.sh: reports in $out (html/index.html)"
