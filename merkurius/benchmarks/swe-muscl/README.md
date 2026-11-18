# Shallow water, second order

`benchmarks/swe` with a heavier scheme: second-order MUSCL-Hancock with
HLLC fluxes instead of first-order Godunov with HLL. The world, the cases,
the tiles and the bookkeeping are the same. A cell's step does several
times the arithmetic, so the fixed costs of starting a tile weigh less, as
they would on a GPU with fast doubles (gfx90a). The first-order programs
stay in `benchmarks/swe` for comparison.

## What it does

The 2D shallow water equations, in depth `h` and momenta `hu` and `hv`,
over a bed shaped like a bowl:

- **Reconstruction (MUSCL):** within each cell, the depth, the surface
  `h + b` and the velocities `u` and `v` are linear, with minmod slopes
  across x and across y. Minmod keeps the faces' depths non-negative, and
  reconstructing the surface keeps a lake at rest flat. A face's bed is its
  surface less its depth.
- **Half step (Hancock):** each face's state moves half a step by the
  differences of the cell's own fluxes at its faces, and by the bed's slope.
  A face at most `1e-6` m deep loses its momenta.
- **Fluxes:** at each face, an HLLC flux (Toro 2001) of the two cells'
  states there, with hydrostatic reconstruction (Audusse et al. 2004). Its
  mass and normal fluxes are HLL's. Its along flux is the mass flux times
  the along velocity of the upwind side of the middle wave.
- **Source:** the bed's slope between a cell's faces, at the half step,
  with Audusse et al.'s second-order term, so that a lake at rest stays at
  rest.
- **Time steps:** each step's `dt` is 0.45 over the largest
  `|u| + |v| + 2 sqrt(g h)` after the step before, as in the first-order
  scheme. Cells are 1 m square, `g` 9.81, no friction.
- **Reach:** a cell's step reads the cells up to two away, across and
  diagonally: 13 cells. A wet cell within two cells of a tile's edge or
  corner starts the tile beyond it.
- **The cases, the bed, the tiles and the output** are `benchmarks/swe`'s.
  `SWE_KERNEL=2` picks `step_faces` in the Mercury and HIP programs. HIP
  has no `step_staged`: a halo of two would need a new staging layout,
  and Mercury has no counterpart to compare it with.

The functions are in `swe.h`:
- `minmod`, the limiter.
- `predict`, a cell's four faces' states half a step on, from it and its
  four neighbours.
- `hllc`, a face's flux.
- `advance`, a cell's new state from its fluxes and its source.

The `step` kernels compute each cell's own faces and its four neighbours'
facing ones, 5 predictions and 4 fluxes a cell. `step_faces` computes each
face's flux once, from its two cells' predictions, into the tile's faces
array in local memory, as in `benchmarks/swe`. The last loop predicts each
cell's own faces again for the source: 5 predictions and 2 fluxes a cell.

## Accuracy

Thacker's case, the relative L1 error of the depth, the steps twice the
world's side so that each world covers the same part of the period:

| World | First order | Second order | Ratio |
|---:|---:|---:|---:|
| 128 | 3.12e-2 | 4.08e-3 | 7.6 |
| 256 | 1.58e-2 | 1.63e-3 | 9.7 |
| 512 | 8.05e-3 | 6.83e-4 | 11.8 |
| 1024 | 4.11e-3 | 3.03e-4 | 13.6 |

The second-order error falls 2.3 to 2.5 times when the cells halve. The
shoreline holds it below 4, since the solution's slope is discontinuous
there. Past a period, with the water at its widest (`SWE_WORLD=128`,
`SWE_TILE=8`, 600 steps), the exact solution has 6836 wet cells. Second
order has 6980, while first order smears thin films over 14496.

Volume is conserved apart from two losses, as in the first-order scheme:
- Thacker's water reaches 0.56 of the world from the centre, so some
  flows out over the world's edge.
- A tile that ends drops its films of at most `1e-6` m. In case 0 at 256
  cells and 1000 steps, that is 0.015 m³ of 3954. With tiles never ending,
  the volume is the same to all 7 digits printed.

## Three programs, the same bits

- `swe-muscl.m`, one GPU process, as `benchmarks/swe/swe-sparse.m`.
  `SWE_KERNEL=2` steps a tile as `step_faces` does, its faces' fluxes in
  group-private arrays (OpenMP 6.1, so `-fopenmp-version=61`).
- `swe-muscl.c`, C with OpenMP offload: a kernel a step, the bookkeeping on
  the host.
- `swe-muscl.hip`, HIP: the same kernels and bookkeeping, a block of 256
  threads a tile. `SWE_KERNEL` 0, the default, is `step`; 2 is
  `step_faces`, at most 32 cells square a tile, otherwise `step`. It also
  prints the step's kernel time alone, by HIP's events.

All three, on the host or a GPU, give the same bits, for the reasons
`benchmarks/swe/README.md` gives: the same operations in the same order,
no fused multiply and add, correctly rounded division and square roots,
and sums whose order does not matter.

## Building and running

`compare-swe-muscl.sh [build-dir] [offload-arch] [rounds]` is
`compare-swe.sh` for these programs:

```sh
python3 -I merkurius/utils/gpu/with-gpus.py --count 1 -- \
  merkurius/benchmarks/swe-muscl/compare-swe-muscl.sh "" gfx1030
```

By hand, as in `benchmarks/swe/README.md`, with `swe-muscl.m`,
`swe-muscl.c` and `swe-muscl.hip`.

## Tests

`test/Benchmarks/swe-muscl.test` runs the Mercury and C programs at the
host on small worlds, `swe-muscl-gpu.test` the Mercury one on the GPU and
`swe-muscl-c-gpu.test` the C and HIP ones. Their outputs are the C
program's at the host, which the three agree with bit for bit.

## Measurements

In `docs/MercuryMLIRCompilerandRuntimeDesign.md`, Measurements, "Shallow
water, second order".
