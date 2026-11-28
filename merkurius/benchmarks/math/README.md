# Math on the GPU

The math library's functions in a GPU loop, to compare the device's math
library that Merkurius links with the one that Clang, HIP and Flang link
(`docs/INPROGRESS.md`, Math on the GPU).

## Where each program's device math comes from

- **Merkurius:** `library/math.m`'s functions are C foreign procs that
  call `<math.h>`, around Mercury's domain checks. Clang compiles them for
  amdgcn as plain calls of `sqrt`, `exp` and so on, and the device link,
  of a GPU process's image or of an OpenMP target region's, resolves them
  from LLVM libc's `libm.a` for amdgcn (`lib/amdgcn-amd-amdhsa/`), which
  LTO inlines.
- **C with OpenMP offload, and HIP:** Clang's wrapper headers map
  `<math.h>` on the device to ROCm's ocml (`__ocml_sin_f64` and so on),
  linked from `ocml.bc`.
- **Flang:** the intrinsics become `math` dialect operations, which
  `ConvertMathToROCDL` makes calls of ocml (flang/lib/Optimizer/CodeGen/
  CodeGen.cpp), linked from `ocml.bc`.

So the three baselines share ocml, and only Merkurius differs.

## What it does

For each of `sqrt`, `exp`, `ln` (`log`), `sin`, `cos`, `pow` and `tanh`,
one OpenMP target loop (`teams distribute parallel do`, a kernel in HIP)
over `MATH_N` elements (2^20). Element `i`, with `c = i / N`, starts at
`c` and applies the function `MATH_K` times (1000), `x = f(x + c)` with
offsets that keep the argument in the function's domain, then adds its
last `x` to the sum. A run of each loop loads its kernel; then the best of
`MATH_REPEAT` runs (5) is timed. Each program prints a line a function:
its name, the best time in seconds and the first run's sum.

Merkurius' functions also check their domain (`sqrt` and `ln` of a
negative number throw), a comparison the others do not make; the loops'
arguments never fail it.

## Running it

    compare-math.sh [build-dir] [offload-arch] [rounds]

builds the four programs at -O2 and runs them in turns, then prints each
function's best time per program and its ratio to HIP's, and how far each
sum is from HIP's: the libraries round differently, and the reductions add
in different orders, so the sums agree only to about the libraries'
accuracy. On a GPU that drives a display it masks half the GPU's units, for
the OpenMP programs and HIP alike, as `pipeline/sweep.sh` does.
