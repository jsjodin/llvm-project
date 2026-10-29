# Merkurius

A Mercury compiler and runtime on MLIR and LLVM. The design is in
[docs/MercuryMLIRCompilerandRuntimeDesign.md](docs/MercuryMLIRCompilerandRuntimeDesign.md).

So far there is the driver, `merkc`, and its frontend, `merkc -fc1`, which
reads source files through Flang's provenance machinery, lexes them, reads
their items as terms, and converts the terms to a parse tree; so far the
converter handles the module system's declarations. Its actions are:

| Action | Prints |
| --- | --- |
| `-E` | The source as the lexer reads it, which is the file unchanged |
| `-fdebug-dump-provenance` | The provenance table of the source files |
| `-fdebug-dump-tokens` | The tokens, one per line |
| `-fdebug-unparse-tokens` | Every token's trivia and spelling, which is the file byte for byte |
| `-fdebug-dump-terms` | Each item's term in canonical notation, or `error` |
| `-fdebug-dump-parse-tree-no-sema` | The parse tree, one node or field per line |

`-fopenmp` makes `%$omp` directives tokens instead of comments.

## Building

The simplest way is `build_merk.sh`, which configures, builds and installs
LLVM, MLIR, Clang, LLD, Flang and Merkurius in one in-tree build, outside the
repository: in `../build-merk`, installed to `../install-merk`, both relative
to `llvm-project`. Run `merkurius/build_merk.sh help` for its commands.

To build by hand: Merkurius needs LLVM and MLIR, and builds in either of two
ways. The commands below run from `llvm-project` and keep the build
directories beside it.

**Standalone,** against an LLVM build or installation that includes MLIR:

```sh
cmake -G Ninja -S llvm -B ../build-llvm \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host
ninja -C ../build-llvm

cmake -G Ninja -S merkurius -B ../build-merkurius \
  -DMLIR_DIR=$PWD/../build-llvm/lib/cmake/mlir
ninja -C ../build-merkurius check-merkurius
```

Against a build tree, `llvm-lit` and `FileCheck` are found there. Against an
installation, build LLVM with `-DLLVM_INSTALL_UTILS=ON` and pass
`-DLLVM_EXTERNAL_LIT=<path to llvm-lit>`.

**In-tree,** as an LLVM external project:

```sh
cmake -G Ninja -S llvm -B ../build \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_EXTERNAL_PROJECTS=merkurius \
  -DLLVM_EXTERNAL_MERKURIUS_SOURCE_DIR=$PWD/merkurius
ninja -C ../build check-merkurius
```

## Testing

`merkurius/build_merk.sh run_tests`, or `ninja check-merkurius` in a build
directory, runs the lit and unit tests. The tests that compare the term reader
with Mercury's own need `mmc` (Mercury 22.01), found through the `MMC`
environment variable or the `PATH`; without it, lit reports them as
unsupported:

```sh
MMC=$HOME/mercury-22.01.9/bin/mmc merkurius/build_merk.sh run_tests
```

`utils/fuzz/run-term-fuzzer.sh [SECONDS] [JOBS]` fuzzes the lexer and term
reader with libFuzzer, using the clang of the build tree.

## Layout

| Directory | Contents |
| --- | --- |
| `include/merkurius`, `lib` | Libraries: `Common` and `Parser` (source files, provenance and diagnostics, adapted from Flang; the lexer, operator table, term reader, converter and parse tree), `Driver` (options), `Frontend`, `FrontendTool` |
| `tools/merkc` | The driver and the `-fc1` entry point |
| `tools/merkurius-term-fuzzer` | A fuzz target for the lexer and term reader |
| `utils/term-oracle` | The term oracle, which reads terms with Mercury's own parser, and the scripts that compare it with merkc and generate test items |
| `utils/fuzz` | The script and dictionary for fuzzing |
| `test` | lit tests; `test/Unit` runs the unit tests |
| `unittests` | googletest unit tests |

Code copied from other projects is listed in [THIRD_PARTY.md](THIRD_PARTY.md).
