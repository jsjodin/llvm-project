#!/usr/bin/env python3
"""Compile programs with mmc and merkc, run them, and compare their output.

Each program is compiled by mmc once, and by merkc at each of the given
optimization levels, and each of merkc's is run once for each heap chunk
size given, with MERK_HEAP_CHUNK_BYTES (runtime/heap.cpp). Every run must
print what mmc's prints, with the same exit status. A program that mmc
cannot compile is reported as "mmc stops", which is the generator's fault;
one that merkc cannot compile, with merkc's first error, which is merkc's
or a construct it has not implemented. Exits with 1 if any differ.

Usage: compare-runs.py --merkc MERKC --mmc MMC [--levels 0 2]
                       [--chunk-bytes 64 ...] [--jobs N] FILE...
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile


def run(command, cwd, env=None, timeout=120):
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                                text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return None, "", "timed out"


def first_difference(expected, actual):
    a, b = expected.splitlines(), actual.splitlines()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return "line %d: mmc %r, merkc %r" % (i + 1, x, y)
    if len(a) != len(b):
        return "mmc %d lines, merkc %d" % (len(a), len(b))
    return None


def compare(path, args):
    """The problems with one program, [] if it agrees, or mmc's first error
    as a string if mmc cannot compile it."""
    # merkc runs in the temporary directory too.
    path = os.path.abspath(path)
    module = os.path.splitext(os.path.basename(path))[0]
    with tempfile.TemporaryDirectory(prefix="compare-runs-") as tmp:
        # mmc wants the file named after the module, in its own directory.
        shutil.copy(path, os.path.join(tmp, module + ".m"))
        # mmc runs its own tools, such as mkinit, from beside it.
        env = dict(os.environ)
        env["PATH"] = os.path.dirname(os.path.abspath(args.mmc)) + os.pathsep + env.get("PATH", "")
        code, out, err = run([args.mmc, "--make", module], tmp, env=env, timeout=600)
        if code != 0:
            lines = (out + err).splitlines()
            return next((l for l in lines if "error" in l), "") or "\n".join(lines[-3:])
        code, expected, _ = run([os.path.join(tmp, module)], tmp)
        problems = []
        for level in args.levels:
            binary = os.path.join(tmp, "merkc-O%s" % level)
            code_c, _, err = run([args.merkc, "-O%s" % level, path, "-o", binary], tmp,
                                 timeout=600)
            if code_c != 0:
                first = next((l for l in err.splitlines() if "error" in l), err.strip())
                problems.append("-O%s: merkc fails: %s" % (level, first))
                continue
            for chunk in [None] + args.chunk_bytes:
                env = dict(os.environ)
                where = "-O%s" % level
                if chunk is not None:
                    env["MERK_HEAP_CHUNK_BYTES"] = str(chunk)
                    where += " MERK_HEAP_CHUNK_BYTES=%s" % chunk
                code_r, actual, err = run([binary], tmp, env=env)
                if code_r != code:
                    problems.append("%s: exit status %s, mmc's %s %s" % (
                        where, code_r, code, err.strip()[:200]))
                elif (difference := first_difference(expected, actual)):
                    problems.append("%s: output differs, %s" % (where, difference))
        return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merkc", required=True)
    parser.add_argument("--mmc", required=True)
    parser.add_argument("--levels", nargs="+", default=["0"])
    parser.add_argument("--chunk-bytes", nargs="*", type=int, default=[])
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    differ = stops = 0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda p: (p, compare(p, args)), args.files))
    for path, problems in results:
        if isinstance(problems, str):
            stops += 1
            print("%s: mmc stops: %s" % (path, problems))
        elif problems:
            differ += 1
            for problem in problems:
                print("%s: %s" % (path, problem))
    print("%d programs, %d differ, %d mmc stops" % (len(args.files), differ, stops))
    sys.exit(1 if differ or stops else 0)


if __name__ == "__main__":
    main()
