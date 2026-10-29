#!/usr/bin/env python3
"""Compare merkc's term reader with Mercury's on the items of Mercury files.

Each file is read by `merkc -fc1 -fdebug-dump-terms` and by the term oracle
(term_oracle.m, built with mmc), which both print one line per item: its
term in canonical notation, or "error". The script exits with status 1 and
reports every item on which they differ.

The oracle is built once into --oracle-dir, and rebuilt when term_oracle.m
changes. A lock lets parallel lit tests share it.
"""

import argparse
import fcntl
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_SOURCE = os.path.join(HERE, "term_oracle.m")


def build_oracle(mmc, oracle_dir):
    os.makedirs(oracle_dir, exist_ok=True)
    oracle = os.path.join(oracle_dir, "term_oracle")
    with open(os.path.join(oracle_dir, "build.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = os.path.join(oracle_dir, "term_oracle.m")
        if (
            not os.path.exists(oracle)
            or not os.path.exists(source)
            or open(source).read() != open(ORACLE_SOURCE).read()
        ):
            shutil.copyfile(ORACLE_SOURCE, source)
            env = dict(os.environ)
            env["PATH"] = os.path.dirname(mmc) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(
                [mmc, "--make", "term_oracle"],
                cwd=oracle_dir,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                sys.exit("building the term oracle failed:\n" + result.stdout + result.stderr)
    return oracle


def item_sources(text):
    """The source of each item, roughly: the text up to each end token."""
    items, start, j = [], 0, 0
    while j < len(text):
        ch = text[j]
        if ch == "%":
            j = text.find("\n", j)
            j = len(text) if j < 0 else j
        elif ch in "\"'":
            j += 1
            while j < len(text) and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            j += 1
            continue
        elif ch == "." and (j + 1 == len(text) or text[j + 1] in " \t\r\n%"):
            if j == 0 or not text[j - 1] in "#$&*+-./:<=>?@^~\\":
                lines = text[start : j + 1].splitlines()
                items.append(" ".join(
                    l.strip() for l in lines if not l.strip().startswith("%")))
                start = j + 1
        j += 1
    return items


def compare(merkc, oracle, path):
    with open(path, "rb") as f:
        data = f.read()
    ours = subprocess.run(
        [merkc, "-fc1", "-fdebug-dump-terms", path], capture_output=True
    ).stdout.decode("utf-8", "replace").splitlines()
    theirs = subprocess.run(
        [oracle], input=data, capture_output=True
    ).stdout.decode("utf-8", "replace").splitlines()
    sources = item_sources(data.decode("utf-8", "replace"))
    differences = 0
    if len(ours) != len(theirs):
        print(f"{path}: merkc read {len(ours)} items, mmc {len(theirs)}")
        differences += 1
    for j, (a, b) in enumerate(zip(ours, theirs)):
        if a != b:
            differences += 1
            source = sources[j] if j < len(sources) else "?"
            print(f"{path}: item {j + 1}: {source}\n  merkc: {a}\n  mmc:   {b}")
            if len(ours) != len(theirs) and differences >= 5:
                print(f"{path}: the items no longer line up; stopping here")
                break
    return differences, len(ours)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merkc", required=True)
    parser.add_argument("--mmc", required=True)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    oracle = build_oracle(args.mmc, args.oracle_dir)
    total_differences = total_items = 0
    for path in args.files:
        differences, items = compare(args.merkc, oracle, path)
        total_differences += differences
        total_items += items
    print(f"{total_items} items, {total_differences} differences")
    sys.exit(1 if total_differences else 0)


if __name__ == "__main__":
    main()
