#!/usr/bin/env bash
# Builds the parser and desugaring fuzz target with libFuzzer and the
# address and undefined behavior sanitizers, using the clang of the build
# tree, then fuzzes for SECONDS (default 300) in JOBS processes (default 40).
# A fifth of the jobs fuzz large inputs, up to 64 KB, which reach what grows
# with the size of a module; the others fuzz inputs up to 4 KB, which run
# many times faster. The two are separate fuzzers that run at once.
#
# SECONDS is the time spent fuzzing, after libFuzzer has run every input
# of the corpus once, which takes minutes for a large corpus: libFuzzer's
# own -max_total_time counts from its start, so each fuzzer runs without
# it and is stopped SECONDS after its first job reports. Each runs in its
# own process group, which is sent SIGTERM, since in fork mode it ignores
# SIGINT and its jobs outlive it, and then SIGKILL for a job it started as
# it stopped. Each reports how long it fuzzed and how many inputs it ran.
# Stopped so, a fuzzer may exit with 72 ("run interrupted") or
# 1, if a job ended as it stopped; the run fails if a fuzzer ended before
# it was stopped with a failure status, or wrote a crash, leak, timeout or
# out-of-memory artifact.
#
# The sources compile in parallel through a generated ninja file, and only
# those that changed since the last run are recompiled. The fuzzers run in
# libFuzzer's fork mode, which keeps a corpus small: it adds an input to
# the corpus only when the input covers something new.
#
# The corpora, crashes, the logs and the fuzzer itself live in
# $BUILD_DIR/fuzz/term: corpus, artifacts and fuzz.log for the small
# inputs, and corpus-large, artifacts-large and fuzz-large.log for the
# large ones. The large inputs have a corpus of their own, so that they do
# not slow the loading of the small one; it starts as a copy of the small
# one. Both are seeded with the Mercury files in the tests. Inputs that
# crash are written as crash-* in the artifacts directories; add them to
# test/Parser/Inputs/fuzz once fixed. Those that take more than 10 seconds
# are written as slow-unit-* and listed at the end, but do not fail the
# run: the instrumented fuzzer is tens of times slower than merkc, so time
# them with merkc.
#
# Usage: utils/fuzz/run-term-fuzzer.sh [SECONDS] [JOBS]

set -euo pipefail

MERKURIUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLVM_PROJECT_DIR="$(dirname "$MERKURIUS_DIR")"
BUILD_DIR="${BUILD_DIR:-$(dirname "$LLVM_PROJECT_DIR")/build-merk}"
SECONDS_TO_RUN="${1:-300}"
JOBS="${2:-40}"
LARGE_JOBS=$((JOBS / 5))
SMALL_JOBS=$((JOBS - LARGE_JOBS))
WORK="$BUILD_DIR/fuzz/term"
mkdir -p "$WORK/corpus" "$WORK/artifacts" "$WORK/artifacts-large"

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

if [[ ! -d "$WORK/corpus-large" ]]; then
  cp -r "$WORK/corpus" "$WORK/corpus-large"
fi
find "$MERKURIUS_DIR/test" "$MERKURIUS_DIR/utils" -name '*.m' \
  -exec cp {} "$WORK/corpus/" \; -exec cp {} "$WORK/corpus-large/" \;

cd "$WORK"
# The fuzzers, by name: their process groups, their logs, the times they
# started fuzzing, once their corpora were loaded, the times they were
# stopped, and the times they ended.
NAMES=(small)
declare -A PID LOG START STOP END
run() {
  local name=$1 jobs=$2 corpus=$3 artifacts=$4 log=$5
  shift 5
  setsid ./merkurius-term-fuzzer "$corpus" -artifact_prefix="$artifacts/" \
    -fork="$jobs" -dict="$MERKURIUS_DIR/utils/fuzz/mercury.dict" "$@" \
    > "$log" 2>&1 &
  PID[$name]=$!
  LOG[$name]=$log
}
run small "$SMALL_JOBS" corpus artifacts fuzz.log -max_len=4096
: > fuzz-large.log
if ((LARGE_JOBS > 0)); then
  NAMES+=(large)
  # Without -len_control=0, libFuzzer would grow its inputs from small
  # ones only slowly.
  run large "$LARGE_JOBS" corpus-large artifacts-large fuzz-large.log \
    -max_len=65536 -len_control=0
fi
tail -n +1 -f fuzz.log fuzz-large.log &
TAIL=$!

running() { kill -0 "${PID[$1]}" 2>/dev/null; }
# A corpus is loaded when the first job reports, as "#N: ... job: 1".
while true; do
  ANY=false
  for name in "${NAMES[@]}"; do
    [[ -z "${END[$name]:-}" ]] || continue
    if ! running "$name"; then
      END[$name]=$SECONDS
      # Any job it started as it ended.
      kill -KILL -- "-${PID[$name]}" 2>/dev/null || true
      continue
    fi
    ANY=true
    if [[ -z "${START[$name]:-}" ]]; then
      if grep -qE '^#[0-9]+: .* job: ' "${LOG[$name]}"; then
        START[$name]=$SECONDS
      fi
    elif [[ -z "${STOP[$name]:-}" ]] &&
        ((SECONDS - START[$name] >= SECONDS_TO_RUN)); then
      STOP[$name]=$SECONDS
      kill -TERM -- "-${PID[$name]}"
    fi
  done
  $ANY || break
  sleep 1
done
sleep 1
kill "$TAIL" 2>/dev/null || true
STATUS=0
for name in "${NAMES[@]}"; do
  FUZZER_STATUS=0
  wait "${PID[$name]}" || FUZZER_STATUS=$?
  if [[ -z "${STOP[$name]:-}" ]]; then
    STOP[$name]=${END[$name]}
    echo "INFO: the $name fuzzer ended before it was stopped," \
      "with exit status $FUZZER_STATUS" | tee -a fuzz.log
    if ((FUZZER_STATUS != 0)); then
      STATUS=1
    fi
  fi
done

for dir in artifacts artifacts-large; do
  SLOW=$(ls "$dir" | grep '^slow-unit-' || true)
  if [[ -n "$SLOW" ]]; then
    echo "INFO: slow inputs in $dir:" $SLOW | tee -a fuzz.log
  fi
  FAILED=$(ls "$dir" | grep -E '^(crash|leak|timeout|oom)-' || true)
  if [[ -n "$FAILED" ]]; then
    echo "INFO: artifacts written in $dir:" $FAILED | tee -a fuzz.log
    STATUS=1
  fi
done
# The inputs run after the corpus was loaded, from the last status line,
# "#N: ... job: ...", whose count starts there.
for name in "${NAMES[@]}"; do
  RUNS=$(grep -E '^#[0-9]+: .* job: ' "${LOG[$name]}" | tail -n 1 |
    sed -E 's/^#([0-9]+):.*/\1/')
  echo "INFO: the $name fuzzer fuzzed for" \
    "$((STOP[$name] - ${START[$name]:-${STOP[$name]}})) seconds" \
    "after loading its corpus, and ran ${RUNS:-0} inputs" | tee -a fuzz.log
done
echo "INFO: exit status $STATUS" | tee -a fuzz.log
exit "$STATUS"
