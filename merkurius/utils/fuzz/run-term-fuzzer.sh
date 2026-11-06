#!/usr/bin/env bash
# Builds the parser and desugaring fuzz target with libFuzzer and the
# address and undefined behavior sanitizers, using the clang of the build
# tree, then runs two fuzzers, one after the other, each for SECONDS
# (default 300) in JOBS processes (default 40): first one of inputs up to
# 4 KB, which run many times faster, then one of large inputs, up to 64 KB,
# which reach what grows with the size of a module. Run one at a time, each
# has every core; the large one found three times the features with 40 jobs
# that it found with 8 while the small one had 32.
#
# SECONDS is the time spent fuzzing, after libFuzzer has run every input
# of the corpus once: libFuzzer's own -max_total_time counts from its start,
# so each fuzzer runs without it and is stopped SECONDS after its first job
# reports. Our libFuzzer runs the corpus in one process per hardware thread
# (-fork_load_jobs, a local change), which takes about a minute for each
# corpus, where upstream's one process took 8 and 23 minutes. Each fuzzer
# runs in its own process group, which is sent SIGTERM, since in fork mode
# it ignores SIGINT and its jobs outlive it, and then SIGKILL for a job it
# started as it stopped. Each reports how long it fuzzed and how many
# inputs it ran.
# Stopped so, a fuzzer may exit with 72 ("run interrupted") or
# 1, if a job ended as it stopped; the run fails if a fuzzer ended before
# it was stopped with a failure status, or wrote a crash, leak, timeout or
# out-of-memory artifact.
#
# The sources compile in parallel through a generated ninja file, and only
# those that changed since the last run are recompiled; libFuzzer's runtime
# is rebuilt first, if it changed, and the fuzzer relinked. The fuzzers run in
# libFuzzer's fork mode, which adds an input to the corpus only when the
# input covers a new feature. Features are many more than edges, so a
# corpus still grows with every run, and so does the time to load it;
# distill-term-corpus.sh, run before fuzzing, distills a corpus that has
# doubled since it was last distilled.
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
WORK="$BUILD_DIR/fuzz/term"
mkdir -p "$WORK/corpus" "$WORK/artifacts" "$WORK/artifacts-large"

# Our libFuzzer, with its local changes, is built with the runtimes.
ninja -C "$BUILD_DIR/runtimes/runtimes-bins" compiler-rt/lib/fuzzer/fuzzer
FUZZER_RUNTIME="$("$BUILD_DIR/bin/clang++" -print-runtime-dir)/libclang_rt.fuzzer.a"
if [[ ! -f "$FUZZER_RUNTIME" ]]; then
  echo "error: libFuzzer's runtime is not at $FUZZER_RUNTIME" >&2
  exit 1
fi

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
  echo "build merkurius-term-fuzzer: link ${OBJECTS[*]} | $FUZZER_RUNTIME"
} > "$WORK/build.ninja"
ninja -C "$WORK" merkurius-term-fuzzer

if [[ ! -d "$WORK/corpus-large" ]]; then
  cp -r "$WORK/corpus" "$WORK/corpus-large"
fi
find "$MERKURIUS_DIR/test" "$MERKURIUS_DIR/utils" -name '*.m' \
  -exec cp {} "$WORK/corpus/" \; -exec cp {} "$WORK/corpus-large/" \;
# A corpus that has doubled since it was last distilled is distilled again,
# so that loading it stays fast.
BUILD_DIR="$BUILD_DIR" "$MERKURIUS_DIR/utils/fuzz/distill-term-corpus.sh"

cd "$WORK"
# The fuzzers, by name: their logs, the times they started fuzzing, once
# their corpora were loaded, the times they were stopped, and the times
# they ended.
NAMES=(small large)
declare -A LOG START STOP END
STATUS=0
: > fuzz.log
: > fuzz-large.log
tail -n +1 -f fuzz.log fuzz-large.log &
TAIL=$!

# Runs one fuzzer, in a process group of its own, until it has fuzzed for
# SECONDS_TO_RUN after its corpus was loaded, or ends.
fuzz() {
  local name=$1 corpus=$2 artifacts=$3 log=$4 pid status=0
  shift 4
  LOG[$name]=$log
  setsid ./merkurius-term-fuzzer "$corpus" -artifact_prefix="$artifacts/" \
    -fork="$JOBS" -dict="$MERKURIUS_DIR/utils/fuzz/mercury.dict" "$@" \
    > "$log" 2>&1 &
  pid=$!
  # A corpus is loaded when the first job reports, as "#N: ... job: 1".
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -z "${START[$name]:-}" ]]; then
      if grep -qE '^#[0-9]+: .* job: ' "$log"; then
        START[$name]=$SECONDS
      fi
    elif [[ -z "${STOP[$name]:-}" ]] &&
        ((SECONDS - START[$name] >= SECONDS_TO_RUN)); then
      STOP[$name]=$SECONDS
      kill -TERM -- "-$pid"
    fi
    sleep 1
  done
  END[$name]=$SECONDS
  # Any job it started as it ended.
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" || status=$?
  if [[ -z "${STOP[$name]:-}" ]]; then
    STOP[$name]=${END[$name]}
    echo "INFO: the $name fuzzer ended before it was stopped," \
      "with exit status $status" | tee -a fuzz.log
    if ((status != 0)); then
      STATUS=1
    fi
  fi
}
fuzz small corpus artifacts fuzz.log -max_len=4096
# Without -len_control=0, libFuzzer would grow its inputs from small ones
# only slowly.
fuzz large corpus-large artifacts-large fuzz-large.log \
  -max_len=65536 -len_control=0
sleep 1
kill "$TAIL" 2>/dev/null || true

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
