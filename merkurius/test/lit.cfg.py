# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "Merkurius"
config.test_format = lit.formats.ShTest()

# Mercury sources, .test files for tests that need no source of their own,
# and MLIR modules in the merk dialect.
config.suffixes = [".m", ".test", ".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.merkurius_obj_root, "test")

# Inputs holds files that tests read. Unit is a separate suite, found through
# its lit.site.cfg.py in the build directory.
config.excludes = ["Inputs", "CMakeLists.txt"]

llvm_config.with_system_environment(["HOME", "TMP", "TEMP"])
# The targets LLVM was built for, as in LLVM's tests: x86-registered-target,
# amdgpu-registered-target and so on.
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + "-registered-target")

llvm_config.use_default_substitutions()

tool_dirs = [config.merkurius_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["merkc", "merkurius-term-fuzzer"], tool_dirs)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# Tests that run target regions on a GPU need one, found as Clang's tests
# find one with amdgpu-arch, and the OpenMP runtime libraries: libomp and
# libomptarget, from the build's lib directories or the runtimes build.
# %offload-arch is the GPU; %libomp-flags links the libraries and
# %libomp-env finds them when the program runs.
import subprocess

build_root = os.path.dirname(config.llvm_tools_dir)
omp_dirs = []
for library in ["libomp.so", "libomptarget.so"]:
    candidates = [os.path.join(build_root, "lib")]
    candidates += [
        os.path.join(build_root, "lib", d)
        for d in sorted(os.listdir(os.path.join(build_root, "lib")))
        if "-linux" in d
    ] if os.path.isdir(os.path.join(build_root, "lib")) else []
    candidates.append(os.path.join(build_root, "runtimes", "runtimes-bins",
                                   "openmp", "runtime", "src"))
    found = [d for d in candidates if os.path.exists(os.path.join(d, library))]
    if found:
        if found[0] not in omp_dirs:
            omp_dirs.append(found[0])
    else:
        omp_dirs = None
        break
amdgpu_arch = os.path.join(config.llvm_tools_dir, "amdgpu-arch")
gpus = []
if os.path.exists(amdgpu_arch):
    try:
        gpus = subprocess.run([amdgpu_arch], capture_output=True, text=True,
                              timeout=60).stdout.split()
    except (OSError, subprocess.SubprocessError):
        gpus = []
# Programs with host parallel regions need only libomp, the "libomp"
# feature.
if omp_dirs:
    config.available_features.add("libomp")
    config.substitutions.append(
        ("%libomp-flags", " ".join("-L" + d for d in omp_dirs)))
    config.substitutions.append(
        ("%libomp-env", "env LD_LIBRARY_PATH=" + ":".join(omp_dirs)))
if gpus and omp_dirs:
    config.available_features.add("amdgpu-gpu")
    config.substitutions.append(("%offload-arch", gpus[0]))

# The unparser's round-trip tests 2, 3 and 4.
config.substitutions.append(
    (
        "%round-trip",
        "%s %s --merkc %s"
        % (
            config.python_executable,
            os.path.join(config.merkurius_src_root, "utils", "round-trip",
                         "round-trip.py"),
            os.path.join(config.merkurius_tools_dir, "merkc"),
        ),
    )
)

# Tests that compare our term reader with Mercury's need mmc, from $MMC or
# the PATH; without it they are unsupported.
import shutil

mmc = os.environ.get("MMC") or shutil.which("mmc")
if mmc and os.path.exists(mmc):
    config.available_features.add("mmc")
    config.substitutions.append(
        (
            "%compare-terms",
            "%s %s --merkc %s --mmc %s --oracle-dir %s"
            % (
                config.python_executable,
                os.path.join(config.merkurius_src_root, "utils", "term-oracle",
                             "compare-terms.py"),
                os.path.join(config.merkurius_tools_dir, "merkc"),
                os.path.abspath(mmc),
                os.path.join(config.merkurius_obj_root, "term-oracle"),
            ),
        )
    )
    # Type checking and inference compared with mmc's (utils/type-oracle).
    config.substitutions.append(
        (
            "%compare-types",
            "%s %s --merkc %s --mmc %s"
            % (
                config.python_executable,
                os.path.join(config.merkurius_src_root, "utils", "type-oracle",
                             "compare-types.py"),
                os.path.join(config.merkurius_tools_dir, "merkc"),
                os.path.abspath(mmc),
            ),
        )
    )
    # Polymorphism analysis compared with mmc's (utils/type-oracle).
    config.substitutions.append(
        (
            "%compare-polymorphism",
            "%s %s --merkc %s --mmc %s"
            % (
                config.python_executable,
                os.path.join(config.merkurius_src_root, "utils", "type-oracle",
                             "compare-polymorphism.py"),
                os.path.join(config.merkurius_tools_dir, "merkc"),
                os.path.abspath(mmc),
            ),
        )
    )
    # Switch detection compared with mmc's (utils/type-oracle).
    config.substitutions.append(
        (
            "%compare-switches",
            "%s %s --merkc %s --mmc %s"
            % (
                config.python_executable,
                os.path.join(config.merkurius_src_root, "utils", "type-oracle",
                             "compare-switches.py"),
                os.path.join(config.merkurius_tools_dir, "merkc"),
                os.path.abspath(mmc),
            ),
        )
    )
