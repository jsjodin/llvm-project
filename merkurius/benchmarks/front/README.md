# A reacting front

The main reference of the demo of a stencil whose active region moves
(`docs/INPROGRESS.md`, step 1): a front that burns through a grid, dense, on
one GPU, written twice. `front.c` is the bar, in C with OpenMP offload, and
`front.m` is the same program in Mercury with Merkurius's OpenMP directives.
Unlike Game of Life (`benchmarks/life`), a burning cell has hundreds of
operations to do each step, and a cold or burnt one almost none, so the work
is heavy where the front is and light elsewhere, as in combustion codes.

## The model

Each cell has a temperature, starting at 1, and fuel, starting at 1, and an
intermediate, starting at 0. A disc of radius 5 cells, a quarter of the way
along each side, starts at temperature 5. The border is cold and stays so.
A step, for each cell:

1. **Diffusion:** the temperature plus 0.2 times its five-point Laplacian,
   from the last step's temperatures; 0.2 is below 0.25, where explicit
   diffusion becomes unstable.
2. **Loss:** 1% of the temperature above 1 goes to the surroundings, so that
   a burnt cell cools down and goes quiet.
3. **Chemistry,** only where the temperature is above 1.5 and fuel or
   intermediate is left: `FRONT_SUBSTEPS` substeps (100 by default) of two
   reactions in turn. Fuel becomes intermediate at a rate of
   `0.5 (T - 1.5)^2` above 1.5, giving heat 1; intermediate becomes product
   at a rate of `2 (T - 2)` above 2, giving heat 3. Each is a linearly
   implicit Euler step, `x / (1 + h k)`, which stays stable however fast the
   rate, as these stiff reactions need. Fuel or intermediate below 10^-9 is
   then gone.

The front moves at about 0.6 cells a step and is about three cells deep.
Behind it the fuel is gone and the cells cool to 1.

- **The kernels** are flat loops over the grid's cells, under
  `target teams distribute parallel do` in Mercury and
  `target teams distribute parallel for` in C, each in a function of its own.
  The substeps are a loop inside, in Mercury a do loop of three `fromto`s.
- **The arrays** are of doubles: the temperature twice, which swap after each
  step, and the fuel and the intermediate, updated in place, as a cell's
  chemistry reads only its own. They stay on the device for all the steps,
  under `target data`.
- **The output** is the number of burnt cells, with less than half their
  fuel left, and the sums of the temperatures and of the fuel, added by rows
  in order, so that every build adds the same way; and each step's time but
  the first, as min, max and average, with cells updated a second from the
  fastest.

## Checking

The model is written a third time, in Python, outside the tree, which gave
the values in the tests. Python, every Mercury build (sequential, on the
host's threads, on the GPU, at -O0 and -O2) and `front.c` on the CPU give
the same bits. `front.c` on the GPU differs in the last digit or so, since
clang fuses multiplies and adds into FMAs on AMD GPUs and merkc does not.

## Building and running

`run.sh [build-dir] [offload-arch]` builds and runs `front.m` sequentially,
on the host's threads and on a GPU. `compare-c.sh [build-dir] [offload-arch]
[rounds]` builds `front.m` with merkc and `front.c` with the same build's
clang, at -O2, runs them in turn for each round, checks that they burn the
same cells and that their sums agree to 12 digits, and prints each one's
median rate and average step:

```sh
python3 -I merkurius/utils/gpu/with-gpus.py --count 1 -- \
  merkurius/benchmarks/front/compare-c.sh "" gfx1030
```

On a machine that others share, both go through `utils/gpu/with-gpus.py`.

## Tests

`test/Benchmarks/front.test` runs `front.m` sequentially and on the host's
threads, at -O0 and -O2; `front-gpu.test` on the GPU, checking that the
kernels are SPMD and that no step copies the arrays; `front-c-gpu.test` the
C program, sequentially to the bit and on the GPU to 14 digits.
