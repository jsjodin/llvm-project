#!/usr/bin/env bash
# Configure, build, install and test Merkurius, built in-tree with LLVM, MLIR,
# Clang, LLD and Flang as an LLVM external project. Run with --help for usage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# The build and install trees sit beside the repository, not inside it.
WORK_DIR="$(dirname "$LLVM_PROJECT_DIR")"
BUILD_DIR="${BUILD_DIR:-$WORK_DIR/build-merk}"
INSTALL_DIR="${INSTALL_DIR:-$WORK_DIR/install-merk}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-40}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [command]

Builds LLVM with MLIR, Clang (with CIR), LLD, Flang and Merkurius, plus the
openmp, compiler-rt and offload runtimes for the host and for AMD GPUs
(amdgcn-amd-amdhsa). Targets: X86, AMDGPU, AArch64. Uses Ninja and ccache.

Commands:
  (none)      Configure from scratch, build and install. Use this the first
              time, or after changing the CMake options in this script.
  configure   Delete the build directory and run CMake again. ccache makes the
              rebuild that follows much faster than the first one.
  build       Build whatever has changed (ninja -j $JOBS).
  install     Build whatever has changed, then install.
  run_tests   Build whatever has changed, then run the Merkurius tests
              (check-merkurius).
  clean       Delete the build directory. The install directory is kept.
  help        Show this message.

Typical use:
  $(basename "$0")              # first time: about 20 minutes on 40 cores
  $(basename "$0") run_tests    # after editing Merkurius sources
  $(basename "$0") install      # update the installed tools

Directories (override with the variables shown):
  Source       $LLVM_PROJECT_DIR
  BUILD_DIR    $BUILD_DIR
  INSTALL_DIR  $INSTALL_DIR

Other settings:
  BUILD_TYPE   $BUILD_TYPE (Release, Debug, RelWithDebInfo; takes effect at
               configure)
  JOBS         $JOBS (parallel compile jobs)

Example: JOBS=16 BUILD_TYPE=Debug $(basename "$0") configure

After installing, put the tools on your PATH:
  export PATH=$INSTALL_DIR/bin:\$PATH
  merkc --help
EOF
}

do_clean() {
  echo "Removing $BUILD_DIR"
  rm -rf "$BUILD_DIR"
}

do_configure() {
  do_clean
  cmake -G Ninja -S "$LLVM_PROJECT_DIR/llvm" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_CXX_STANDARD=17 \
    -DBUILD_SHARED_LIBS=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;lld;flang" \
    -DLLVM_EXTERNAL_PROJECTS=merkurius \
    -DLLVM_EXTERNAL_MERKURIUS_SOURCE_DIR="$SCRIPT_DIR" \
    -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU;AArch64" \
    -DLLVM_ENABLE_RUNTIMES="openmp;compiler-rt;offload" \
    -DLLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa" \
    -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=openmp \
    -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON \
    -DCOMPILER_RT_BUILD_SANITIZERS=ON \
    -DCLANG_DEFAULT_LINKER=lld \
    -DCLANG_ENABLE_CIR=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_PARALLEL_LINK_JOBS=8
}

require_configured() {
  if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
    echo "error: $BUILD_DIR is not configured; run '$0 configure' first" >&2
    exit 1
  fi
}

do_build() {
  require_configured
  ninja -C "$BUILD_DIR" -j "$JOBS"
}

do_install() {
  require_configured
  # The install prefix is fixed in the CMake cache; update it in place if
  # INSTALL_DIR has changed, rather than configuring from scratch.
  local prefix
  prefix="$(sed -n 's/^CMAKE_INSTALL_PREFIX:PATH=//p' "$BUILD_DIR/CMakeCache.txt")"
  if [[ "$prefix" != "$INSTALL_DIR" ]]; then
    echo "Changing the install prefix from $prefix to $INSTALL_DIR"
    cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" "$BUILD_DIR"
  fi
  # The install target builds whatever is out of date first.
  ninja -C "$BUILD_DIR" -j "$JOBS" install
}

do_run_tests() {
  require_configured
  ninja -C "$BUILD_DIR" -j "$JOBS" check-merkurius
}

case "${1:-all}" in
  all)
    do_configure
    do_build
    do_install
    ;;
  configure) do_configure ;;
  build) do_build ;;
  install) do_install ;;
  run_tests) do_run_tests ;;
  clean) do_clean ;;
  -h|--help|help) usage ;;
  *)
    echo "error: unknown command '$1'" >&2
    echo "Run '$(basename "$0") help' for usage." >&2
    exit 1
    ;;
esac
