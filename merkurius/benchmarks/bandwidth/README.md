# Memory bandwidth

`bandwidth.m` measures memory bandwidth in the manner of
[BabelStream](https://github.com/UoB-HPC/BabelStream) (Deakin et al.,
"Evaluating attainable memory bandwidth of parallel programming models via
BabelStream", 2018), written in Mercury with Merkurius's OpenMP directives.
It is our own program, not a port of BabelStream's code. Its numbers are
not BabelStream benchmark results, and are comparable with BabelStream's
only as far as its kernels and its measurement are the same, below; when
publishing them, call them results of a benchmark modeled on BabelStream.

## What it does

Three arrays of `float` (double precision), `A`, `B` and `C`, start at 0.1,
0.2 and 0.0. Five kernels run in turn, `BANDWIDTH_NUM_TIMES` times (100
by default), over `BANDWIDTH_ARRAY_SIZE` elements (2^25 by default, 268 MB
an array):

| Kernel | Operation | Bytes moved |
| --- | --- | --- |
| Copy | `C = A` | 2 × 8 × N |
| Mul | `B = 0.4 × C` | 2 × 8 × N |
| Add | `C = A + B` | 3 × 8 × N |
| Triad | `A = B + 0.4 × C` | 3 × 8 × N |
| Dot | `Sum = A · B` | 2 × 8 × N |

Each kernel is a do loop under `%$omp target teams distribute parallel do`,
and Dot has `reduction((+) : Sum)`. The arrays are threaded through the
kernels as unique arrays, updated in place; a kernel reads the others as
`array_ui`. The program times each kernel with `clock_gettime`, and leaves
out each kernel's first run, as BabelStream does. Its bandwidth is from its
fastest run, in MB/s (10^6 bytes).

Before it prints the table, it checks the arrays as BabelStream does: it
runs the kernels on scalars for the expected value of each element. Each
array's average error, relative to that value, must be within 100 times
the machine epsilon, and the dot product's within 10^-8. If a check fails,
it says which, and exits with status 1.

## Building and running

`run.sh [build-dir] [offload-arch]` builds and runs the program three ways.
The build directory defaults to `build-merk`, beside `llvm-project`:

```sh
merkurius/benchmarks/bandwidth/run.sh "" gfx1030
```

- **Sequential**, `merkc -O2 -fdo-loops`: the directives are comments.
- **OpenMP on the host**, `merkc -O2 -fopenmp -fdo-loops`: with no device
  image, OpenMP runs each target region on the host, its teams and threads
  on the host's cores. `OMP_NUM_THREADS` sets the threads.
- **OpenMP on a GPU**, `merkc -O2 -fopenmp -fdo-loops --offload-arch=gfx1030`,
  run with `OMP_TARGET_OFFLOAD=MANDATORY`. Each kernel is an SPMD kernel.

The OpenMP builds link libomp and libomptarget from the build; the script
sets `-L` and `LD_LIBRARY_PATH` for them.

## Results

On an AMD Ryzen Threadripper PRO 5965WX (24 cores, 48 threads) and an AMD
Radeon RX 6950 XT (gfx1030), with the default sizes, on 2026-11-03:

| Kernel | Sequential | OpenMP, host (48 threads) | OpenMP, GPU |
| --- | --- | --- | --- |
| Copy | 28,480 MB/s | 101,136 MB/s | 10,458 MB/s |
| Mul | 28,509 MB/s | 99,230 MB/s | 10,456 MB/s |
| Add | 31,994 MB/s | 109,234 MB/s | 12,759 MB/s |
| Triad | 30,957 MB/s | 108,234 MB/s | 12,741 MB/s |
| Dot | 20,610 MB/s | 140,469 MB/s | 22,999 MB/s |

**The GPU column measures transfers, not the GPU's memory.** Merkurius has
no `target data` or `target enter data` yet, so each kernel's target region
maps the arrays it uses, by their modes: the arrays it reads `to` the
device, and the one it updates `tofrom`. Copy, for example, moves 268 MB to
the device and 536 MB to and from it, over PCIe, in each run, which is its
51 ms. BabelStream's OpenMP model keeps the arrays on the device with
`target enter data` before the runs, so its numbers are the device's memory
bandwidth. That needs data regions in Merkurius; until then, the GPU
build checks that the kernels and the reduction are right on the device.

## Differences from BabelStream

- One program for every build: BabelStream's OpenMP model chooses between
  `target teams distribute parallel for` and `parallel for` when it is
  compiled; ours runs the target directive's host fallback on the host.
- Its settings come from the environment rather than the command line,
  since Merkurius's library has no lists yet, which
  `io.command_line_arguments` needs.
- No `nstream` kernel, and only double precision.
- The arrays are initialized by a kernel of the same kind, as BabelStream
  initializes them, so that, on the host, each thread first touches the
  elements it uses.

`test/Benchmarks/bandwidth.test` builds it and runs it on 100003 elements,
sequentially and on four host threads; `bandwidth-gpu.test` runs it on the
GPU.
