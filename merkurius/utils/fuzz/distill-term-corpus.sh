#!/usr/bin/env bash
# Distills the term fuzzer's corpora: keeps, of each, a smallest set of
# inputs that covers all that the corpus covers, using libFuzzer's
# -merge=1. A fork mode fuzzer adds an input to its corpus whenever it
# covers a new feature, and features are many more than edges, so a corpus
# grows with every run, and so does the time to load it: every input is run
# once before fuzzing starts, though our libFuzzer runs them in one process
# per hardware thread. A distilled corpus covers the same with fewer
# inputs, so it loads faster: the first distillation kept about two thirds
# of the small corpus and three quarters of the large one.
#
# Usage: distill-term-corpus.sh [--force]
#
# A corpus is distilled when it has grown to GROWTH (default 2) times its
# size after it was last distilled, or has never been distilled, or with
# --force. run-term-fuzzer.sh runs this before it fuzzes; it can also be
# run on its own once the fuzzer is built. The small and the large corpus
# are distilled at once, each with its fuzzer's -max_len.
#
# A corpus is distilled into a new directory, which then replaces it; the
# corpus it replaced is kept as, for example, corpus.prev, until the next
# distillation replaces that. The size after distilling is kept in, for
# example, corpus.distilled, and the merge's output in, for example,
# distill-corpus.log.
set -euo pipefail

MERKURIUS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLVM_PROJECT_DIR="$(dirname "$MERKURIUS_DIR")"
BUILD_DIR="${BUILD_DIR:-$(dirname "$LLVM_PROJECT_DIR")/build-merk}"
WORK="$BUILD_DIR/fuzz/term"
GROWTH="${GROWTH:-2}"
FORCE=false
if [[ "${1:-}" == "--force" ]]; then
  FORCE=true
fi

cd "$WORK"
if [[ ! -x ./merkurius-term-fuzzer ]]; then
  echo "error: $WORK/merkurius-term-fuzzer is not built;" \
    "run run-term-fuzzer.sh first" >&2
  exit 1
fi

# Distills one corpus, if it needs it, and reports what it did.
distill() {
  local corpus=$1 max_len=$2
  [[ -d "$corpus" ]] || return 0
  local size last
  size=$(find "$corpus" -maxdepth 1 -type f | wc -l)
  last=$(cat "$corpus.distilled" 2>/dev/null || echo 0)
  if ! $FORCE && ((last > 0 && size < GROWTH * last)); then
    echo "INFO: $corpus has $size inputs, $last after it was last" \
      "distilled; not distilled"
    return 0
  fi
  local new="$corpus.new"
  rm -rf "$new"
  mkdir "$new"
  if ! ./merkurius-term-fuzzer -merge=1 -max_len="$max_len" "$new" \
      "$corpus" > "distill-$corpus.log" 2>&1; then
    echo "error: distilling $corpus failed; see $WORK/distill-$corpus.log" >&2
    rm -rf "$new"
    return 1
  fi
  local kept
  kept=$(find "$new" -maxdepth 1 -type f | wc -l)
  rm -rf "$corpus.prev"
  mv "$corpus" "$corpus.prev"
  mv "$new" "$corpus"
  echo "$kept" > "$corpus.distilled"
  echo "INFO: distilled $corpus from $size inputs to $kept;" \
    "the old corpus is $corpus.prev"
}

distill corpus 4096 &
SMALL=$!
distill corpus-large 65536 &
LARGE=$!
STATUS=0
wait "$SMALL" || STATUS=1
wait "$LARGE" || STATUS=1
exit "$STATUS"
