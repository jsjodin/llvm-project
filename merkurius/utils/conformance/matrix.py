#!/usr/bin/env python3
"""Map the reference manual's sections to the tests that test them.

Each test may name what it tests in its header: `% MANUAL: <node>`, a node
of the reference manual as manual-sections.txt lists them, or `% DESIGN:
<section>`, a section of the design document (docs/INPROGRESS.md, A
conformance suite). This prints the manual's sections in order, each with
its conformance tests by class (the start of a test's name in
test/Conformance) and its other tests, and marks a section that has
subsections of its own with none, or no run- or reject- test, as a gap; then
the design document's sections that tests name. It also checks that a
conformance test is in test/Conformance/<chapter>/<section>/, by the slugs
of its chapter and its section, and exits with 1 if a test names a node
the manual does not have or is in the wrong directory. The column `known'
counts the tests with a `% KNOWN-BUG:' line, which are expected to fail
(oracle.py); --known lists them with their reasons instead.

  matrix.py [--gaps | --known] [--sections FILE] [TEST-DIR...]
  matrix.py --extract-sections mercury_ref.info > manual-sections.txt
"""

import argparse
import collections
import glob
import os
import re
import sys

CLASSES = ("run", "reject", "warn", "nyi", "differs")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def extract_sections(info):
    """The manual's nodes in order, with their depths, from its info files."""
    text = "".join(open(f, encoding="latin-1").read()
                   for f in sorted(glob.glob(info + "*")))
    up = {}
    order = []
    for chunk in text.split("\x1f"):
        m = re.search(r"Node: ([^,\n]*)", chunk)
        if not m:
            continue
        node = m.group(1).strip()
        u = re.search(r"Up: ([^,\n]*)", chunk)
        up[node] = u.group(1).strip() if u else ""
        order.append(node)
    children = collections.defaultdict(list)
    for node in order:
        children[up[node]].append(node)
    lines = []

    def walk(node, depth):
        for child in children[node]:
            # Its entries are references, not rules.
            if child == "Bibliography":
                continue
            lines.append("%d|%s" % (depth, child))
            walk(child, depth + 1)
    walk("Top", 0)
    return lines


def read_sections(path):
    sections = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            depth, name = line.rstrip("\n").split("|", 1)
            sections.append((int(depth), name))
    return sections


def chapters_of(sections):
    chapter = {}
    current = None
    for depth, name in sections:
        if depth == 0:
            current = name
        chapter[name] = current
    return chapter


def scan(dirs):
    """Each test's names, with its class, and whether it is a conformance
    test."""
    for top in dirs:
        for root, subdirs, files in os.walk(top):
            subdirs[:] = sorted(d for d in subdirs
                                if d not in ("Inputs", "Output"))
            for name in sorted(files):
                if not name.endswith((".m", ".mt", ".test")):
                    continue
                path = os.path.join(root, name)
                manual, design, known = [], [], []
                with open(path, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.startswith(("%", "#")):
                            break
                        m = re.match(r"[%#] MANUAL:\s*(.*\S)", line)
                        if m:
                            manual.append(m.group(1))
                        m = re.match(r"[%#] DESIGN:\s*(.*\S)", line)
                        if m:
                            design.append(m.group(1))
                        m = re.match(r"[%#] KNOWN-BUG:\s*(.*\S)", line)
                        if m:
                            known.append(m.group(1))
                conformance = "/Conformance/" in os.path.abspath(path).replace(
                    os.sep, "/")
                kind = name.split("-", 1)[0] if conformance else "other"
                yield path, kind, manual, design, known, conformance


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("dirs", nargs="*",
                        default=[os.path.join(here, "..", "..", "test")])
    parser.add_argument("--sections",
                        default=os.path.join(here, "manual-sections.txt"))
    parser.add_argument("--gaps", action="store_true",
                        help="print only the sections that are gaps")
    parser.add_argument("--known", action="store_true",
                        help="list the tests with known bugs, and why")
    parser.add_argument("--extract-sections", metavar="INFO")
    args = parser.parse_args()
    if args.extract_sections:
        print("\n".join(extract_sections(args.extract_sections)))
        return
    sections = read_sections(args.sections)
    nodes = {name for _, name in sections}
    chapter = chapters_of(sections)
    counts = collections.defaultdict(collections.Counter)
    designs = collections.defaultdict(collections.Counter)
    problems = []
    knowns = []
    for path, kind, manual, design, known, conformance in scan(args.dirs):
        for reason in known:
            knowns.append((path, reason))
        for name in manual:
            if name not in nodes:
                problems.append("%s: no manual node `%s'" % (path, name))
                continue
            counts[name][kind] += 1
            counts[name]["known"] += bool(known)
            parts = path.replace(os.sep, "/").split("/")
            # A test of a section is in its chapter's directory, in the
            # section's; one of a chapter without sections, in the
            # chapter's.
            want = [slug(chapter[name])]
            if name != chapter[name]:
                want.append(slug(name))
            if conformance and parts[-1 - len(want):-1] != want:
                problems.append("%s: belongs in Conformance/%s/" % (
                    path, "/".join(want)))
        for name in design:
            designs[name][kind] += 1
            designs[name]["known"] += bool(known)
    if args.known:
        for path, reason in knowns:
            print("%s: %s" % (os.path.relpath(path), reason))
        print("%d known bugs" % len(knowns))
        sys.exit(0)
    has_children = {name for (d, name), (d2, _) in
                    zip(sections, sections[1:] + [(0, "")]) if d2 > d}
    columns = CLASSES + ("other", "known")
    print("%-58s %s" % ("Manual section", " ".join("%7s" % c for c in columns)))
    gaps = 0
    for depth, name in sections:
        c = counts[name]
        leaf = name not in has_children
        gap = leaf and c["run"] + c["reject"] == 0
        gaps += gap
        if args.gaps and not gap:
            continue
        label = ("  " * depth + name)[:56]
        cells = " ".join("%7s" % (c[k] or ".") for k in columns)
        print("%-56s %s %s" % (label, "!" if gap else " ", cells))
    print("\n%d of %d sections without subsections have no run- or reject- "
          "test" % (gaps, sum(1 for _, n in sections if n not in has_children)))
    if designs:
        print("\n%-58s %s" % ("Design section",
                              " ".join("%7s" % c for c in columns)))
        for name in sorted(designs):
            print("%-58s %s" % (name[:58], " ".join(
                "%7s" % (designs[name][k] or ".") for k in columns)))
    for problem in problems:
        print(problem, file=sys.stderr)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
