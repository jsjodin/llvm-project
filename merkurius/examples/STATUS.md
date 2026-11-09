# Merkurius: status, November 2026

Merkurius is a clean-room Mercury compiler built on MLIR and LLVM. It is written
from the Mercury 22.01 manuals, published papers and experiments with mmc's
output, never from Melbourne Mercury's source. The aim is to compile standard
Mercury to native code at C speed, and later to add parallel loops and
Erlang-style processes. The programs in this directory run today: `make` lists
the commands.

## Done

**Front end.** A lexer and parser for the whole language, including OpenMP-style
`%$omp` directives. The reader agrees with mmc's term reader, and is fuzzed
after every change.

**Semantic analysis, complete and compatible with mmc.** Every pass is checked
against mmc 22.01.9 by oracles that generate random programs. About 13,000
generated modules agree with mmc on every error line and determinism.
- Name resolution and module imports.
- Type checking and inference, type classes, functional dependencies,
  existential types, subtypes and `coerce`.
- Purity checking.
- Mode analysis, after Overton's thesis: constrained polymorphic modes, unique
  modes, and mode inference.
- Determinism analysis: switch detection, committed choice, inference.
- Uniqueness checking. It is sound by default, and `-fmmc-uniqueness` gives
  mmc's rules.
- Polymorphism analysis: where each `type_info` and `typeclass_info` comes from.

mmc bugs found on the way are recorded with reproducers, so they can be reported.

**Code generation.**
- **Path:** Mercury → our `merk` MLIR dialect → MLIR's LLVM dialect → LLVM IR
  → native code. A Clang-style driver (`merkc -c`, `-O2`, `--target=`) links
  the result with our runtime.
- **Covered so far:**
  - `det` and `semidet` procedures on primitive types;
  - integer and float arithmetic, with mmc's checks and error messages;
  - if-then-else, switches, negation;
  - tail calls guaranteed with LLVM's `musttail`, so loops run in constant
    stack.
- **Foreign code:** `pragma foreign_proc("C")` is compiled by clang and linked
  into the module as bitcode, so LLVM can inline it.
- **Library:** the `io` output predicates and the `float` and `math` modules
  work.
- **Target:** the target is recorded on the MLIR module, so a dumped `.mlir`
  file can be compiled again later.

**Testing.** 378 lit tests check both the MLIR and the LLVM IR, and the
programs' output against mmc's. There are unit tests, the random-program
sweeps, and a fuzzer.

## What remains

| Area | Work |
| --- | --- |
| Data | Term layouts and the `mrt` runtime dialect: lists, discriminated unions, strings, construction and switches on constructors |
| Control | Disjunctions and `nondet` code, by continuation passing; commits |
| Higher order | `type_info` passing, type-class dispatch, closures and higher-order calls |
| Exceptions | `throw` and `catch` with LLVM's zero-cost exception handling. Now an exception ends the program, as mmc's report does |
| Memory | A precise, moving garbage collector with LLVM statepoints, and per-process heaps |
| Library | The rest of the standard library (`list`, `string`, `map`, ...). Later, `math` through MLIR's `math` dialect and LLVM intrinsics |
| Modules | Interface files and separate compilation, with `foreign_decl` visible to importers |
| Extensions | Parallel loops lowered to OpenMP, lightweight processes with mailboxes, open terms, and offloading to GPUs |
| Tooling | Debug information, and performance gates against mmc's measured baseline |

The order planned next: terms and the runtime's allocation, then nondeterminism
and higher-order code. That is when ordinary Mercury programs, such as list
processing, start to compile.
