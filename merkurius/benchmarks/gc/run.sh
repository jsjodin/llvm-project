#!/usr/bin/env bash
# Builds the collector's benchmarks and runs each on one pinned core five
# times, printing the median elapsed and user seconds and peak memory
# (README.md).
#
#   run.sh [build-dir] [core]
#
# build-dir is the Merkurius build, ../../../../build-merk by default; core
# is the CPU to pin to, 2.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
core="${2:-2}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
for program in nrev deep live; do
  "$build/bin/merkc" -O2 "$here/$program.m" -o "$out/$program"
done

# The run whose user time is the median of five: elapsed, user, peak KB.
median() {
  local times="$out/times"
  : > "$times"
  for _ in 1 2 3 4 5; do
    /usr/bin/time -f "%e %U %M" -o "$out/time" \
      taskset -c "$core" "$@" > /dev/null
    cat "$out/time" >> "$times"
  done
  sort -k2 -n "$times" | sed -n 3p
}

printf '%-36s %8s %8s %10s\n' program elapsed user "peak KB"
run() {
  printf '%-36s %8s %8s %10s\n' "$*" $(median "$out/$1" "${@:2}")
}
run nrev 3001 500
run deep 300000 5000000
run live 8000000 10000000
