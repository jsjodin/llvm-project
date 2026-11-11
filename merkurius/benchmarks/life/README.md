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

## Tests

`test/Benchmarks/life.test` runs `life.m` sequentially and on the host's
threads, at -O0 and -O2; `life-gpu.test` on the GPU, checking that the
kernels are SPMD and that no step copies the arrays; `life-c-gpu.test` the C
program with both cell types. All check the population and checksum, which
come from an independent program in Python.
