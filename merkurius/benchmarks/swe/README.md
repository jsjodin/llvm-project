# Shallow water

A sparse computation with much more work per cell than Life
(`docs/INPROGRESS.md`, A showcase with more work per tile). It is the
showcase for distributing tiles across places: Life's tile is about 0.65 µs
of a GPU's time a step, against a place's 200 to 450 µs of agreement a step,
so distributing it cannot pay. Here water sloshes across a dry bowl and
only the tiles where it is are stepped, as `life-sparse.m` steps only the
tiles where life is.

## What it does

The 2D shallow water equations, in depth `h` and momenta `hu` and `hv`,
over a bed shaped like a bowl:

- **The scheme** is first-order Godunov finite volumes: at each face of a
  cell, an HLL flux of the states either side, with hydrostatic
  reconstruction (Audusse et al. 2004), so that a lake at rest stays at
  rest and depths stay non-negative at the shoreline. Time steps are
  explicit Euler, each step's `dt` 0.45 over the largest
  `|u| + |v| + 2 sqrt(g h)` after the step before. Cells are 1 m square,
  `g` 9.81, no friction.
- **Wetting and drying:** a cell at most `1e-6` m deep is dry, its momenta
  zero. The wet region moves over the dry bed, and the tiles follow it.
- **The bed** is a paraboloid, `b = D0 (r^2 / R^2 - 1)`, `D0` 10 m, `R`
  0.45 of the world, at the world's centre, computed wherever it is needed.
- **Two cases**, `SWE_CASE`: 0, the showcase, a column of water 5 m deep
  with a flat top, of radius a sixteenth of the world, a fifth of the world
  east of the centre, which collapses and sloshes across the bowl; 1,
  Thacker's oscillating paraboloid (1981), whose exact solution has a moving
  shoreline, for correctness: the programs print the relative L1 error of
  the depth. It falls by half when the cells halve, as a first-order scheme
  should: 2.90e-2, 1.48e-2, 7.45e-3 and 3.74e-3 for worlds of 128 to 1024
  cells over the same time.
- **Tiles** are `life-sparse.m`'s: a pool of `SWE_CAPACITY` tiles of
  `SWE_TILE` cells square, each `3 T^2` doubles, its `h` by rows, then its
  `hu`, then its `hv`; a directory of the world's places; nine neighbours a
  tile, tile 0 the dry tile. A wet cell on a tile's edge or corner starts
  the tile beyond it; a tile with no wet cell and no wet edge facing it for
  4 steps ends.
- **The output** is the simulated time, the wet cells, the volume, a
  checksum of every active cell's bits, the error for case 1, and the active
  tiles; then each step's time but the first as min, max and average, with
  the bookkeeping's part of the average. `SWE_TRACE=1` prints each step's
  line first, with its active tiles.

The settings are `SWE_WORLD` (2048 cells square), `SWE_TILE` (32),
`SWE_STEPS` (1000), `SWE_CASE` (0) and `SWE_CAPACITY` (the side in tiles
squared, at most 4096, plus the dry tile).

## Three programs, the same bits

- `swe-sparse.m`, one GPU process: a step is a loop under `distribute` over
  the active tiles, a team each, with a loop under `parallel do` over the
  tile's cells across the team's threads, which reduces its wet cells, its
  wet edges and its largest speed. The process's lane does the bookkeeping,
  as `life-sparse.m`'s.
- `swe-sparse.c`, C with OpenMP offload: a kernel a step, the bookkeeping on
  the host, as `life-sparse.c`.
- `swe-sparse.hip`, HIP: the same kernels and bookkeeping, a block of 256
  threads a tile, its reductions in shared memory; the bar a GPU programmer
  would set.

The C and HIP programs share the scheme and the host's bookkeeping in
`swe.h`. All three, on the host or a GPU, give the same bits: every
operation is in the same order, nothing is contracted into a fused multiply
and add (merkc emits none; C and HIP are built with `-ffp-contract=off`),
division and square roots are correctly rounded, and `dt` comes from a
maximum, which no order changes. The checksum is the sum, modulo 2^64, over
the active cells of `(bits(h) + 3 bits(hu) + 5 bits(hv)) (2 g + 1)`, `g` the
cell's index in the world, which no order changes either. The volume and
the error are sums whose order differs, printed to 6 and 4 digits.

## Building and running

`compare-swe.sh [build-dir] [offload-arch] [rounds]` builds the three at
-O2 with the same build's merkc and clang, and ROCm's HIP runtime, runs them
in turn for each round, checks that they agree, and prints each one's
median average step and bookkeeping:

```sh
python3 -I merkurius/utils/gpu/with-gpus.py --count 1 -- \
  merkurius/benchmarks/swe/compare-swe.sh "" gfx1030
```

`ROCM_PATH` is `/opt/rocm` by default. By hand:

```sh
merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 \
  $LIBS swe-sparse.m -o swe-m
clang -O2 -ffp-contract=off -fopenmp --offload-arch=gfx1030 $LIBS \
  swe-sparse.c -lm -o swe-c
clang -x hip -O2 -ffp-contract=off --offload-arch=gfx1030 \
  --rocm-path=/opt/rocm swe-sparse.hip -L/opt/rocm/lib -lamdhip64 -lm \
  -Wl,-rpath,/opt/rocm/lib -o swe-hip
```

`$LIBS` is `-L` for the build's `lib` directories and the OpenMP runtime's,
as `compare-swe.sh` has them. `swe-m host` spawns the process at the host.

## Tests

`test/Benchmarks/swe-sparse.test` runs the Mercury and C programs at the
host on small worlds, `swe-sparse-gpu.test` the Mercury one on the GPU and
`swe-sparse-c-gpu.test` the C and HIP ones. Their outputs are the C
program's at the host, which the three agree with bit for bit.
