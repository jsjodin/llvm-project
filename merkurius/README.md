# Merkurius

A Mercury compiler and runtime on MLIR and LLVM. The design is in
[docs/MercuryMLIRCompilerandRuntimeDesign.md](docs/MercuryMLIRCompilerandRuntimeDesign.md).

So far there is the driver, `merkc`, and its frontend, `merkc -fc1`, which reads
source files through Flang's provenance machinery. Its actions are `-E`, which
prints the source as the lexer will read it (the file unchanged), and
`-fdebug-dump-provenance`.

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

## Layout

| Directory | Contents |
| --- | --- |
| `include/merkurius`, `lib` | Libraries: `Common` and `Parser` (source files and provenance, both adapted from Flang), `Driver` (options), `Frontend`, `FrontendTool` |
| `tools/merkc` | The driver and the `-fc1` entry point |
| `test` | lit tests; `test/Unit` runs the unit tests |
| `unittests` | googletest unit tests |

Code copied from other projects is listed in [THIRD_PARTY.md](THIRD_PARTY.md).
