# Game of Life

The first reference of the demo of a stencil whose active region moves
(`docs/INPROGRESS.md`, step 1): Conway's Game of Life, dense, on one GPU,
written twice. `life.c` is the bar, in C with OpenMP offload, and `life.m`
is the same program in Mercury with Merkurius's OpenMP directives. Together
they say whether Mercury's code is slower than C's before any process is
involved, which the process-oriented version would have to make up for.

## What it does

The grid is `LIFE_SIZE` cells square (4096 by default), with a border of
dead cells around it, in one array of `(LIFE_SIZE + 2)^2` cells by rows.
About 30% of the cells start alive, chosen by a multiplicative hash of each
cell's index, so every program starts from the same grid without a random
number generator. A step computes every cell from its eight neighbours into
the other array: a cell with three live neighbours lives, one with two keeps
its state, and any other dies. The two arrays swap after each step, for
`LIFE_STEPS` steps (100 by default).

- **The kernels** are flat loops over the grid's cells, under
  `target teams distribute parallel do` in Mercury and
  `target teams distribute parallel for` in C, each in a function of its own,
  computing a cell's row and column from its index.
- **The arrays stay on the device** for all the steps, under
  `target data map(tofrom: A, B)`, so that a step measures its kernel and not
  copies.
- **The cells** are Mercury's `int`, 8 bytes. The C program's cell type is
  `CELL`: `int64_t` by default, to compare like with like, and `uint8_t`
  (`-DCELL=uint8_t`) as C would usually have it, which shows what Mercury's
  wider cells cost. Narrow fields are planned with flat records
  (`docs/INPROGRESS.md`).
- **The output** is the final population and a checksum, the sum of the live
  cells' indices, which every build must agree on, as Life is exact; and each
  step's time but the first, which includes starting the kernels, as min,
  max and average, with cells updated a second from the fastest.

## Building and running

`run.sh [build-dir] [offload-arch]` builds and runs `life.m` sequentially,
on the host's threads and on a GPU. `compare-c.sh [build-dir] [offload-arch]
[rounds]` builds `life.m` with merkc and `life.c` with the same build's clang,
with both cell types, all at -O2, runs them in turn for each round, checks
that they agree, and prints each one's median rate and average step:

```sh
python3 -I merkurius/utils/gpu/with-gpus.py --count 1 -- \
  merkurius/benchmarks/life/compare-c.sh "" gfx1030
```

On a machine that others share, both go through `utils/gpu/with-gpus.py`,
which gives them a GPU that no one else is using.

## Two nodes over TCP

`life_two.m` is step 2 of the demo: the same Life on two nodes on one
machine, over TCP on loopback, both with the GPU. Node 1 owns the top half
of the rows, node 2 the bottom half, each in an array with a halo row above
and below. After each step, a target region on the GPU makes an array of the
node's edge row, with the step's number after it, sends it to the other
node's mailbox, receives the other's and copies it into the halo. Node 2
sends its population and checksum to node 1 at the end, which prints the
whole grid's, and its step times, with the exchanges' part of them.

```sh
python3 -I merkurius/benchmarks/life/pair.py PROGRAM
```

`pair.py` starts node 2, `PROGRAM partner`, which prints its port, and then
node 1, `PROGRAM first PORT`, with a cookie of their own and a temporary
HOME. `LIFE_SIZE` and `LIFE_STEPS` pass through.

2026-11-11, gfx1030, 300 x 300 cells: an exchange from the GPU took about
240 µs, from the CPU (the sequential build) about 16 µs. A probe put most of
it on the region's arena (`docs/INPROGRESS.md`, the demo's step 2), which a
fix to the runtime is next for; this is not yet a measurement of the design.

## In a GPU process

`life-process.m` runs the same steps in a GPU process, each a loop across
the process's teams under `%$omp distribute parallel do`, rather than a
target region each (`docs/INPROGRESS.md`, GPU processes that run loops).
The process makes both grids in its arena, times each step by the device's
clock, counts the live cells with a loop of two reductions, and sends the
times and the count to the host, which prints them as `life.m` does.
`compare-c.sh` runs it as a fourth program.

2026-11-12, gfx1030, on compute units 0-39 (`LIBOMPTARGET_AMDGPU_CU_MASK`),
the median over the rounds of each program's fastest step:

| Program | 4096², 100 steps | 1024², 200 steps |
| --- | ---: | ---: |
| Mercury, target regions | 21.7 Gcells/s | 27.3 Gcells/s |
| Mercury, GPU process | 24.6 Gcells/s | 29.8 Gcells/s |
| C, `int64_t` cells | 20.5 Gcells/s | 23.7 Gcells/s |
| C, `uint8_t` cells | 31.7 Gcells/s | 23.3 Gcells/s |

A step of the GPU process starts no kernel, which counts most where steps
are short. The others time their steps from the host, a target region's
start included.

## Tests

`test/Benchmarks/life.test` runs `life.m` sequentially and on the host's
threads, at -O0 and -O2; `life-gpu.test` on the GPU, checking that the
kernels are SPMD and that no step copies the arrays; `life-c-gpu.test` the C
program with both cell types; `life-two.test` and `life-two-gpu.test` the two
nodes, with even and odd splits; `life-process-gpu.test` the GPU process,
in parallel and on the process's own lane. All check the population and checksum, which
come from an independent program in Python.
