#!/usr/bin/env bash
# Builds the parser and desugaring fuzz target with libFuzzer and the
# address and undefined behavior sanitizers, using the clang of the build
# tree, then fuzzes for SECONDS (default 180) in JOBS processes (default 32).
# SECONDS is the time spent fuzzing, after libFuzzer has run every input
# of the corpus once, which takes minutes for a large corpus: libFuzzer's
# own -max_total_time counts from its start, so the fuzzer runs without
# it and is stopped SECONDS after its first job reports. It runs in its own
# process group, which is sent SIGTERM, since in fork mode it ignores
# SIGINT and its jobs outlive it.
#
# The sources compile in parallel through a generated ninja file, and only
# those that changed since the last run are recompiled. The fuzzer runs in
# libFuzzer's fork mode, which keeps the corpus small: it adds an input to
# the corpus only when the input covers something new.
#
# The corpus, crashes, the log and the fuzzer itself live in
# $BUILD_DIR/fuzz/term. The corpus is seeded with the Mercury files in the
# tests. Inputs that crash are written there as artifacts/crash-*; add them
# to test/Parser/Inputs/fuzz once fixed. The log is fuzz.log.
#
# Usage: utils/fuzz/run-term-fuzzer.sh [SECONDS] [JOBS]

set -euo pipefail

MERKURIUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLVM_PROJECT_DIR="$(dirname "$MERKURIUS_DIR")"
BUILD_DIR="${BUILD_DIR:-$(dirname "$LLVM_PROJECT_DIR")/build-merk}"
SECONDS_TO_RUN="${1:-180}"
JOBS="${2:-32}"
WORK="$BUILD_DIR/fuzz/term"
mkdir -p "$WORK/corpus" "$WORK/artifacts"

SOURCES=("$MERKURIUS_DIR/tools/merkurius-term-fuzzer/merkurius-term-fuzzer.cpp"
  "$MERKURIUS_DIR"/lib/Parser/*.cpp "$MERKURIUS_DIR"/lib/Semantics/*.cpp
  "$MERKURIUS_DIR"/lib/Common/*.cpp)
{
  cat <<NINJA
cxx = $BUILD_DIR/bin/clang++
sanitize = -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined
cflags = -std=c++17 -O1 -g \$sanitize -DMERKURIUS_LIBFUZZER \
  -I $MERKURIUS_DIR/include -I $LLVM_PROJECT_DIR/llvm/include \
  -I $BUILD_DIR/include
ldflags = -L $BUILD_DIR/lib -Wl,-rpath,$BUILD_DIR/lib \
  -lLLVMFrontendOpenMP -lLLVMSupport -lpthread
rule cxx
  command = \$cxx -MMD -MF \$out.d \$cflags -c \$in -o \$out
  depfile = \$out.d
  deps = gcc
  description = Compiling \$in
rule link
  command = \$cxx \$sanitize \$in \$ldflags -o \$out
  description = Linking \$out
NINJA
  OBJECTS=()
  for source in "${SOURCES[@]}"; do
    object="obj/$(basename "$(dirname "$source")")-$(basename "$source" .cpp).o"
    OBJECTS+=("$object")
    echo "build $object: cxx $source"
  done
  echo "build merkurius-term-fuzzer: link ${OBJECTS[*]}"
} > "$WORK/build.ninja"
ninja -C "$WORK" merkurius-term-fuzzer

find "$MERKURIUS_DIR/test" "$MERKURIUS_DIR/utils" -name '*.m' \
  -exec cp {} "$WORK/corpus/" \;

cd "$WORK"
setsid ./merkurius-term-fuzzer corpus -artifact_prefix=artifacts/ -fork="$JOBS" \
  -max_len=4096 -dict="$MERKURIUS_DIR/utils/fuzz/mercury.dict" \
  > fuzz.log 2>&1 &
FUZZER=$!
tail -n +1 -f --pid="$FUZZER" fuzz.log &

# The corpus is loaded when the first job reports, as "#N: ... job: 1".
until grep -qE '^#[0-9]+: .* job: ' fuzz.log || ! kill -0 "$FUZZER" 2>/dev/null; do
  sleep 1
done
START=$SECONDS
while kill -0 "$FUZZER" 2>/dev/null && ((SECONDS - START < SECONDS_TO_RUN)); do
  sleep 1
done
STOPPED=false
if kill -0 "$FUZZER" 2>/dev/null; then
  STOPPED=true
  kill -TERM -- "-$FUZZER"
fi
STATUS=0
wait "$FUZZER" || STATUS=$?
# Stopped by SIGTERM, libFuzzer exits with 72 ("run interrupted").
if $STOPPED && ((STATUS == 72)); then
  STATUS=0
fi
sleep 1
echo "INFO: fuzzed for $((SECONDS - START)) seconds after loading the corpus;" \
  "exit status $STATUS" | tee -a fuzz.log
exit "$STATUS"
