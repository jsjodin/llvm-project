# The collector

Three programs that each stress a different part of the copying collector
(`runtime/roots.cpp`; design document, Measurements, "The collector"):

- `nrev N R`: naive reverse of a list of N ints, R times. Almost nothing
  survives a collection.
- `deep D R`: R lists of 100 ints made and dropped at the bottom of a
  recursion D frames deep. Each collection walks every frame.
- `live N R`: R lists of 100 ints made and dropped beside a live list of N
  ints, 16 bytes a cell. Each collection copies the live list.

`run.sh [build-dir] [core]` builds them and runs each five times on one
pinned core: nrev 3001 × 500, deep 300,000 × 5,000,000, live 8,000,000 ×
10,000,000, a live list of 128 MB. It prints the median run's elapsed and
user seconds and peak memory.

The programs that measured the collector first, on 2026-11-08, were not
kept; these were written from their descriptions in the design document,
on 2026-11-22, so their times are not comparable with those.
