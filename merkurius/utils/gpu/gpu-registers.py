#!/usr/bin/env python3
"""Reports the registers, stack and spills of a program's GPU code.

For each GPU image of a program, its kernels with their occupancy, and the
functions that need the most vector registers, with what each needs itself
and with its callees, its scratch, and its spills and reloads. It builds
the program for one GPU or more without running anything, so it needs no
GPU, and reports for any GPU the build knows (docs/INPROGRESS.md, Register
pressure in GPU processes):

  python3 -I utils/gpu/gpu-registers.py --arch gfx90a benchmarks/swe-muscl/swe-muscl.m
  python3 -I utils/gpu/gpu-registers.py --arch gfx90a --arch gfx1030 \\
      benchmarks/swe-muscl/swe-muscl.m benchmarks/swe-muscl/swe-muscl.hip

The inputs:
- a Mercury source, built by merkc with --merkc-flags (by default the GPU
  benchmarks' -O2 -fopenmp -fopenmp-version=61 -fdo-loops
  -fprocess-declarations) and -save-temps, which writes each GPU process
  image's assembly (<stem>-<arch>-processes.lto.s) and OpenMP's device
  image's (*.img.lto.s);
- a HIP source (.hip), its device code compiled to assembly at -O2 with
  -ffp-contract=off, as the benchmarks build it;
- a C source (.c), with OpenMP offload, as the benchmarks build it.

The numbers are LLVM's own: the assembly is assembled again with
llvm-mc --save-temp-labels, which resolves each function's register
counts (.L<function>.num_vgpr and so on, which are the most of the
function's and its callees'), and each kernel's occupancy expression, copied
into a symbol of its own. A function's own registers are the first operand
of its count's max(). Spills and reloads are counted from the backend's
comments on them, statically: where they are in loops is not weighed.
--json prints the same as JSON, for tests.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUILD = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..",
                                              "build-merk"))
DEFAULT_MERKC_FLAGS = ("-O2 -fopenmp -fopenmp-version=61 -fdo-loops "
                       "-fprocess-declarations")


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).splitlines()
        sys.exit("gpu-registers: failed: %s\n%s" %
                 (" ".join(command), "\n".join(output[:20])))
    return result.stdout


def library_flags(build):
    dirs = [os.path.join(build, "runtimes", "runtimes-bins", "openmp",
                         "runtime", "src"),
            os.path.join(build, "lib"),
            os.path.join(build, "lib", "x86_64-unknown-linux-gnu")]
    return ["-L" + d for d in dirs]


def build_mercury(args, source, work):
    out = os.path.join(work, "prog")
    command = [os.path.join(args.build, "bin", "merkc")]
    command += args.merkc_flags.split()
    command += ["--offload-arch=" + ",".join(args.arch), "-save-temps=obj"]
    command += library_flags(args.build) + [source, "-o", out]
    run(command, cwd=work)
    images = []
    for path in sorted(glob.glob(os.path.join(work, "*-processes.lto.s"))):
        arch = re.search(r"-(gfx[0-9a-z]+)-processes\.lto\.s$", path).group(1)
        images.append(("GPU processes", arch, path))
    for path in sorted(glob.glob(os.path.join(work, "*.img.lto.s"))):
        arch = re.search(r"\.(gfx[0-9a-z]+)\.img\.lto\.s$", path).group(1)
        images.append(("OpenMP offload", arch, path))
    return images


def build_hip(args, source, work):
    images = []
    for arch in args.arch:
        out = os.path.join(work, "hip-%s.s" % arch)
        run([os.path.join(args.build, "bin", "clang"), "-x", "hip", "-O2",
             "-ffp-contract=off", "--offload-arch=" + arch,
             "--rocm-path=" + args.rocm_path, "--offload-device-only", "-S",
             source, "-o", out], cwd=work)
        images.append(("HIP", arch, out))
    return images


def build_c(args, source, work):
    out = os.path.join(work, "c")
    command = [os.path.join(args.build, "bin", "clang"), "-O2",
               "-ffp-contract=off", "-fopenmp",
               "--offload-arch=" + ",".join(args.arch), "-save-temps=obj"]
    command += library_flags(args.build) + [source, "-lm", "-o", out]
    run(command, cwd=work)
    images = []
    for path in sorted(glob.glob(os.path.join(work, "*.img.lto.s"))):
        arch = re.search(r"\.(gfx[0-9a-z]+)\.img\.lto\.s$", path).group(1)
        images.append(("OpenMP offload", arch, path))
    return images


SPILL = re.compile(r";.*\b(Folded )?Spill\b")
RELOAD = re.compile(r";.*\b(Folded )?Reload\b")


def parse_assembly(path):
    """The functions of an assembly file, in order: each one's name, whether
    it is a kernel, its occupancy expression if it is one, its own vector
    registers, and its spills, reloads and code bytes."""
    functions = {}
    order = []
    current = None
    own = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*\.type\s+([^,\s]+),\s*@function", line)
            if m:
                current = m.group(1)
                if current not in functions:
                    functions[current] = {"kernel": False, "spills": 0,
                                          "reloads": 0, "bytes": 0,
                                          "occupancy_expr": None}
                    order.append(current)
                continue
            m = re.match(r"\s*\.set\s+(\S+)\.num_vgpr,\s*(.*)$", line)
            if m:
                value = m.group(2).strip()
                first = re.match(r"max\(\s*(\d+)", value)
                if first:
                    own[m.group(1)] = int(first.group(1))
                elif value.isdigit():
                    own[m.group(1)] = int(value)
                continue
            if current is None:
                continue
            info = functions[current]
            if SPILL.search(line):
                info["spills"] += 1
            elif RELOAD.search(line):
                info["reloads"] += 1
            m = re.match(r"; codeLenInByte = (\d+)", line)
            if m:
                info["bytes"] = int(m.group(1))
            m = re.match(r"; Occupancy: (.*)$", line)
            if m:
                info["kernel"] = True
                info["occupancy_expr"] = m.group(1).strip()
            if re.match(r"\s*\.amdhsa_kernel\s", line):
                info["kernel"] = True
    for name in order:
        functions[name]["own_vgpr"] = own.get(
            name, own.get(".L" + name))
    return order, functions


def resolve(args, arch, path, order, functions, work):
    """Each function's resolved counts and each kernel's occupancy, by
    assembling the file again with its temporary labels kept."""
    with open(path) as f:
        text = f.read()
    cpu = arch
    extra = []
    for i, name in enumerate(order):
        expr = functions[name]["occupancy_expr"]
        if expr is not None:
            extra.append(".set merk_gpu_registers_occupancy_%d, %s" % (i, expr))
    copy = os.path.join(work, os.path.basename(path) + ".resolve.s")
    with open(copy, "w") as f:
        f.write(text)
        f.write("\n" + "\n".join(extra) + "\n")
    obj = copy + ".o"
    run([os.path.join(args.build, "bin", "llvm-mc"), "-triple=amdgcn-amd-amdhsa",
         "-mcpu=" + cpu, "-filetype=obj", "--save-temp-labels", copy, "-o",
         obj])
    symbols = {}
    for line in run([os.path.join(args.build, "bin", "llvm-nm"), obj]).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in ("a", "A"):
            symbols[parts[2]] = int(parts[0], 16)

    def symbol(name, field):
        for prefix in (".L", ""):
            key = "%s%s.%s" % (prefix, name, field)
            if key in symbols:
                return symbols[key]
        return None

    for i, name in enumerate(order):
        info = functions[name]
        info["vgpr"] = symbol(name, "num_vgpr")
        info["agpr"] = symbol(name, "num_agpr")
        info["sgpr"] = symbol(name, "numbered_sgpr")
        info["scratch"] = symbol(name, "private_seg_size")
        info["occupancy"] = symbols.get("merk_gpu_registers_occupancy_%d" % i)
    return cpu


def demangle(args, names):
    tool = os.path.join(args.build, "bin", "llvm-cxxfilt")
    if not names or not os.path.exists(tool):
        return {n: n for n in names}
    out = subprocess.run([tool], input="\n".join(names), capture_output=True,
                         text=True).stdout.splitlines()
    return dict(zip(names, out)) if len(out) == len(names) else {
        n: n for n in names}


def short(name, width):
    """A demangled name without its parameters, cut to width."""
    if name.endswith(")"):
        depth = 0
        for i in range(len(name) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(name[i], 0)
            if depth == 0:
                name = name[:i]
                break
    return name if len(name) <= width else name[:width - 3] + "..."


def number(value):
    return "-" if value is None else str(value)


def report(args, kind, arch, cpu, path, order, functions):
    names = demangle(args, order)
    kernels = [n for n in order if functions[n]["kernel"]]
    others = [n for n in order if not functions[n]["kernel"]]
    others.sort(key=lambda n: (functions[n].get("vgpr") or 0,
                               functions[n].get("own_vgpr") or 0),
                reverse=True)
    if args.function:
        pattern = re.compile(args.function)
        others = [n for n in others if pattern.search(names[n])]
    else:
        others = others[:args.top]
    print("== %s, %s: %s" % (kind, cpu or arch, path))
    if not order:
        print("No functions.\n")
        return
    print("%-44s %6s %6s %6s %8s %5s %7s %7s" %
          ("Kernel", "VGPR", "AGPR", "SGPR", "Scratch", "Waves", "Spills",
           "Reloads"))
    for n in kernels:
        f = functions[n]
        print("%-44s %6s %6s %6s %8s %5s %7d %7d" %
              (short(names[n], 44), number(f["vgpr"]), number(f["agpr"]),
               number(f["sgpr"]), number(f["scratch"]),
               number(f["occupancy"]), f["spills"], f["reloads"]))
    print("%-44s %6s %6s %6s %8s %7s %7s %7s" %
          ("Function", "VGPR", "Own", "AGPR", "Scratch", "Spills", "Reloads",
           "Bytes"))
    for n in others:
        f = functions[n]
        print("%-44s %6s %6s %6s %8s %7d %7d %7d" %
              (short(names[n], 44), number(f["vgpr"]), number(f["own_vgpr"]),
               number(f["agpr"]), number(f["scratch"]), f["spills"],
               f["reloads"], f["bytes"]))
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+",
                        help="Mercury (.m), HIP (.hip) or C (.c) sources")
    parser.add_argument("--arch", action="append",
                        help="a GPU to build for, such as gfx90a; repeatable "
                             "(default gfx90a)")
    parser.add_argument("--build", default=DEFAULT_BUILD,
                        help="the Merkurius build (default %(default)s)")
    parser.add_argument("--merkc-flags", default=DEFAULT_MERKC_FLAGS,
                        help="merkc's flags (default %(default)s)")
    parser.add_argument("--rocm-path",
                        default=os.environ.get("ROCM_PATH", "/opt/rocm"))
    parser.add_argument("--top", type=int, default=12,
                        help="functions shown an image (default %(default)s)")
    parser.add_argument("--function",
                        help="show the functions whose names match this "
                             "regular expression instead of the top ones")
    parser.add_argument("--json", action="store_true",
                        help="print JSON, every function, instead of tables")
    parser.add_argument("--keep", help="build in this directory and keep it")
    args = parser.parse_args()
    args.arch = args.arch or ["gfx90a"]
    args.build = os.path.abspath(args.build)

    results = []
    for source in args.sources:
        source = os.path.abspath(source)
        if args.keep:
            work = os.path.join(os.path.abspath(args.keep),
                                os.path.basename(source) + ".d")
            os.makedirs(work, exist_ok=True)
        else:
            work = tempfile.mkdtemp(prefix="gpu-registers-")
        try:
            if source.endswith(".m"):
                images = build_mercury(args, source, work)
            elif source.endswith(".hip"):
                images = build_hip(args, source, work)
            elif source.endswith(".c"):
                images = build_c(args, source, work)
            else:
                sys.exit("gpu-registers: %s: expected a .m, .hip or .c source"
                         % source)
            if not images:
                sys.exit("gpu-registers: %s: no GPU image's assembly found"
                         % source)
            for kind, arch, path in images:
                order, functions = parse_assembly(path)
                cpu = resolve(args, arch, path, order, functions, work)
                if args.json:
                    names = demangle(args, order)
                    results.append({
                        "source": source, "image": kind, "arch": cpu or arch,
                        "functions": [dict(functions[n], name=n,
                                           demangled=names[n])
                                      for n in order]})
                else:
                    report(args, kind, arch, cpu,
                           os.path.basename(path) if not args.keep else path,
                           order, functions)
        finally:
            if not args.keep:
                shutil.rmtree(work, ignore_errors=True)
    if args.json:
        for r in results:
            for f in r["functions"]:
                f.pop("occupancy_expr", None)
        json.dump(results, sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
