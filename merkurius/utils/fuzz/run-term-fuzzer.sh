#!/usr/bin/env bash
# Builds the lexer and term reader fuzz target with libFuzzer and the
# address and undefined behavior sanitizers, using the clang of the build
# tree, then fuzzes for SECONDS (default 300) on JOBS processes (default 8).
#
# The corpus, crashes and the fuzzer itself live in $BUILD_DIR/fuzz/term. The
# corpus is seeded with the Mercury files in the tests. Inputs that crash are
# written there as crash-*; add them to test/Parser/Inputs/fuzz once fixed.
#
# Usage: utils/fuzz/run-term-fuzzer.sh [SECONDS] [JOBS]

set -euo pipefail

MERKURIUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLVM_PROJECT_DIR="$(dirname "$MERKURIUS_DIR")"
BUILD_DIR="${BUILD_DIR:-$(dirname "$LLVM_PROJECT_DIR")/build-merk}"
SECONDS_TO_RUN="${1:-300}"
JOBS="${2:-8}"
WORK="$BUILD_DIR/fuzz/term"
mkdir -p "$WORK/corpus" "$WORK/artifacts"

"$BUILD_DIR/bin/clang++" -std=c++17 -O1 -g \
  -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined \
  -DMERKURIUS_LIBFUZZER \
  -I "$MERKURIUS_DIR/include" -I "$LLVM_PROJECT_DIR/llvm/include" \
  -I "$BUILD_DIR/include" \
  "$MERKURIUS_DIR/tools/merkurius-term-fuzzer/merkurius-term-fuzzer.cpp" \
  "$MERKURIUS_DIR"/lib/Parser/*.cpp "$MERKURIUS_DIR"/lib/Common/*.cpp \
  -L "$BUILD_DIR/lib" -Wl,-rpath,"$BUILD_DIR/lib" -lLLVMSupport \
  -lpthread -o "$WORK/merkurius-term-fuzzer"

find "$MERKURIUS_DIR/test" "$MERKURIUS_DIR/utils" -name '*.m' \
  -exec cp {} "$WORK/corpus/" \;

cd "$WORK"
exec ./merkurius-term-fuzzer corpus -artifact_prefix=artifacts/ \
  -max_total_time="$SECONDS_TO_RUN" -jobs="$JOBS" -workers="$JOBS" \
  -max_len=4096 -dict="$MERKURIUS_DIR/utils/fuzz/mercury.dict"
