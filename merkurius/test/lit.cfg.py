# -*- Python -*-

import os
import sys

import lit.formats
from lit.llvm import llvm_config

# What lit's workers must import is in utils/lit/merkurius_lit.py.
sys.path.insert(0, os.path.join(config.merkurius_src_root, "utils", "lit"))
import merkurius_lit

config.name = "Merkurius"
# GpuShTest once the GPUs are known, below.
config.test_format = lit.formats.ShTest()

# Merkurius sources, .test files for tests that need no source of their own,
# and MLIR modules in the merk dialect.
config.suffixes = [".m", ".test", ".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.merkurius_obj_root, "test")

# Inputs holds files that tests read. Unit is a separate suite, found through
# its lit.site.cfg.py in the build directory.
config.excludes = ["Inputs", "CMakeLists.txt"]

# MERK_HEAP_CHUNK_BYTES makes every program's heap chunks that size
# (runtime/heap.cpp): MERK_HEAP_CHUNK_BYTES=64 llvm-lit runs every program
# with nearly every allocation on the slow path. MERK_GPU_PEER=1 makes GPU
# processes send directly to those on other GPUs (runtime/gpu-processes.cpp,
# PeerWanted). ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES choose the GPUs
# the tests may use, as utils/gpu/with-gpus.py sets them: on a machine that
# others share, a test must not reach the GPUs they use.
llvm_config.with_system_environment(
    ["HOME", "TMP", "TEMP", "MERK_HEAP_CHUNK_BYTES", "MERK_GPU_PEER",
     "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"])
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
# %offload-archs is every GPU that the machines the tests run on have, so
# that a test which builds a program and runs it gets an image for this
# machine's GPU, whichever it is: the runtime picks the image by name. Add a
# GPU here to test on a machine with it. First in the list, since lit
# substitutes in order and %offload-arch is a prefix of it.
config.substitutions.insert(0, ("%offload-archs", "gfx1030,gfx90a"))
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
# The compute units that GPU processes take in the tests:
# LIBOMPTARGET_AMDGPU_CU_MASK if the environment sets it, else half of the
# GPU's if it drives a display, so that the display keeps the rest, as
# gfx1030's 40 of 80, else all of them, as on a server's gfx90a.
visible_gpus = merkurius_lit.visible_gpus() if gpus else []
unit_mask = os.environ.get("LIBOMPTARGET_AMDGPU_CU_MASK",
                           merkurius_lit.gpu_unit_mask(visible_gpus))
if omp_dirs:
    config.available_features.add("libomp")
    config.substitutions.append(
        ("%libomp-flags", " ".join("-L" + d for d in omp_dirs)))
    config.substitutions.append(
        ("%libomp-env", "env LD_LIBRARY_PATH=" + ":".join(omp_dirs)))
    # %gpu-env is %libomp-env for programs that spawn GPU processes, whose
    # kernel takes every compute unit that the offload plugin's queues may
    # use (runtime/compute-units.h), limited to unit_mask's.
    config.substitutions.append(
        ("%gpu-env", "env LD_LIBRARY_PATH=" + ":".join(omp_dirs) +
         ("" if unit_mask is None else
          " LIBOMPTARGET_AMDGPU_CU_MASK=" + unit_mask)))
if gpus and omp_dirs:
    config.available_features.add("amdgpu-gpu")
    config.substitutions.append(("%offload-arch", gpus[0]))
    # %one-unit-mask is the fewest compute units that a test can give the
    # offload plugin's queues (LIBOMPTARGET_AMDGPU_CU_MASK) on every GPU it
    # may run on: one, or a pair on a processor with work group processors,
    # which enables units in pairs.
    config.substitutions.append(
        ("%one-unit-mask", merkurius_lit.one_unit_mask(visible_gpus)))
# HIP programs, the baselines of some benchmarks, need ROCm's HIP runtime,
# from $ROCM_PATH or /opt/rocm: the "hip" feature, with %rocm-path for the
# build and %hip-env to run, which limits HSA's queues to the compute units
# that %gpu-env gives GPU processes, as HSA_CU_MASK.
rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
if gpus and os.path.exists(os.path.join(rocm_path, "lib", "libamdhip64.so")):
    config.available_features.add("hip")
    config.substitutions.append(("%rocm-path", rocm_path))
    config.substitutions.append(
        ("%hip-env",
         "env" if unit_mask is None else "env HSA_CU_MASK=0:" + unit_mask))


# The GPU tests, the -gpu.test files, run at most MERK_LIT_GPU_JOBS at a
# time, 4 unless the environment says otherwise, spread over the GPUs that
# the tests see, each test on one (GpuShTest): as many on a GPU as fit
# evenly, 4 on a machine with one GPU, one each on a machine with 4 or
# more. With one per lit worker on one GPU, the GPU had more programs'
# queues than its scheduler holds ("Runlist is getting oversubscribed" in
# the kernel's log), so it took queues off the GPU in turns: a target teams
# region's later teams then did not start for seconds and the region was
# reported, and once a queue could not be taken off in time and the driver
# reset the GPU, which also drives the display. On gfx90a four at once on
# one GPU still failed now and then, kernels whose teams must all run at
# once not getting them beside the others'.
gpu_jobs = int(os.environ.get("MERK_LIT_GPU_JOBS", "4"))
config.parallelism_group = merkurius_lit.gpu_parallelism_group
if visible_gpus:
    per_gpu = max(1, -(-gpu_jobs // len(visible_gpus)))
    gpu_ids = [gpu["id"] for gpu in visible_gpus]
    # More GPUs than jobs: use only as many GPUs as jobs.
    gpu_ids = gpu_ids[:gpu_jobs] if len(gpu_ids) > gpu_jobs else gpu_ids
    config.test_format = merkurius_lit.GpuShTest(
        gpu_ids, per_gpu, os.path.join(config.test_exec_root, "gpu-slots"))
    gpu_jobs = len(gpu_ids) * per_gpu
    # The -two-gpus.test files run on two of them (GpuShTest), with GPU
    # processes on each sending directly to the other's where the GPUs
    # reach each other's memory, as every pair of r11's does.
    if len(gpu_ids) >= 2 and "amdgpu-gpu" in config.available_features:
        config.available_features.add("two-gpus")
lit_config.parallelism_groups["merkurius-gpu"] = gpu_jobs

# The script that runs a command on free GPUs (utils/gpu/with-gpus.py).
config.substitutions.append(
    (
        "%with-gpus",
        "%s -I %s"
        % (
            config.python_executable,
            os.path.join(config.merkurius_src_root, "utils", "gpu",
                         "with-gpus.py"),
        ),
    )
)

# The static report of GPU code's registers (utils/gpu/gpu-registers.py), of
# this build.
config.substitutions.append(
    (
        "%gpu-registers",
        "%s -I %s --build %s"
        % (
            config.python_executable,
            os.path.join(config.merkurius_src_root, "utils", "gpu",
                         "gpu-registers.py"),
            os.path.dirname(config.llvm_tools_dir),
        ),
    )
)

# The driver that runs a distribution test's nodes against each other
# (utils/conformance/nodes.py).
config.substitutions.append(
    (
        "%nodes",
        "%s -I %s"
        % (
            config.python_executable,
            os.path.join(config.merkurius_src_root, "utils", "conformance",
                         "nodes.py"),
        ),
    )
)

# The harness that kills nodes (utils/faults/kill-nodes.py).
config.substitutions.append(
    (
        "%kill-nodes",
        "%s -I %s"
        % (
            config.python_executable,
            os.path.join(config.merkurius_src_root, "utils", "faults",
                         "kill-nodes.py"),
        ),
    )
)

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

# Tests that compare our term reader with mmc's need mmc, from $MMC or
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
    # Programs run and their output compared with mmc's programs'
    # (utils/codegen-oracle).
    config.substitutions.append(
        (
            "%compare-runs",
            "%s %s --merkc %s --mmc %s"
            % (
                config.python_executable,
                os.path.join(config.merkurius_src_root, "utils", "codegen-oracle",
                             "compare-runs.py"),
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

# Tests that run the compiler under valgrind, for reads of memory it has
# freed, which change its output only by chance; without valgrind they are
# unsupported.
valgrind = shutil.which("valgrind")
if valgrind:
    config.available_features.add("valgrind")
    config.substitutions.append(
        ("%valgrind", "%s -q --error-exitcode=99" % valgrind))
