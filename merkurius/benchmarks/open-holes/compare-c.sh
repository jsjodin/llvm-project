#!/usr/bin/env bash
# Builds open-holes.m and ground-holes.m, and E5's C model
# (experiments/open-holes), and runs each on one pinned core five times at
# E5's size, printing the median seconds of each (README.md).
#
#   compare-c.sh [build-dir] [N] [R] [core]
#
# build-dir is the Merkurius build, ../../../../build-merk by default; the
# list has N elements, 1000000 by default, built and summed R times, 200;
# core is the CPU to pin to, 2.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="$(cd "${1:-$here/../../../../build-merk}" && pwd)"
n="${2:-1000000}"
r="${3:-200}"
core="${4:-2}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$build/bin/merkc" -O2 -fopen-types "$here/open-holes.m" -o "$out/open-holes"
"$build/bin/merkc" -O2 "$here/ground-holes.m" -o "$out/ground-holes"
"$build/bin/clang" -O2 -march=native -Wall -Wextra \
  "$here/../../experiments/open-holes/model.c" -o "$out/model"

# The median of five lines of numbers, column by column.
median() {
  sort -n | sed -n 3p
}

printf '%-28s %10s %10s\n' "program" "elapsed" "user"
for program in open-holes ground-holes; do
  : > "$out/times"
  for _ in 1 2 3 4 5; do
    /usr/bin/time -f "%e %U" -o "$out/time" \
      taskset -c "$core" "$out/$program" "$n" "$r" > "$out/output"
    cat "$out/time" >> "$out/times"
  done
  elapsed="$(cut -d' ' -f1 "$out/times" | median)"
  user="$(cut -d' ' -f2 "$out/times" | median)"
  printf '%-28s %10s %10s   %s\n' "merkurius $program" "$elapsed" "$user" \
    "$(cat "$out/output")"
done
for row in ground-btf ground-ftb hole hole-trail; do
  : > "$out/times"
  for _ in 1 2 3 4 5; do
    taskset -c "$core" "$out/model" "$row" "$n" "$r" > "$out/output"
    cut -f4 "$out/output" >> "$out/times"
  done
  printf '%-28s %10s %10s   sum %s\n' "model $row" \
    "$(median < "$out/times")" "" "$(cut -f5 "$out/output")"
done
