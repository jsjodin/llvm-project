#!/usr/bin/env python3
"""The Mandelbrot zoom of mandel.m and mandel.c, independently, for checking
their output on small frames (docs/INPROGRESS.md, step 4 of the demo).

Python's floats are IEEE doubles and it fuses no multiply and add, so it
computes the same iterations as the programs, which follow the same order of
operations. MANDEL_SIZE, MANDEL_FRAMES and MANDEL_TRACE as theirs; it prints
the lines that they print but the times.
"""
import os

X0 = -0.743643887037151
Y0 = 0.131825904205330


def setting(name, fallback):
    s = os.environ.get(name, "")
    return int(s) if s else fallback


def escape(cx, cy, limit):
    """The iterations before z escapes, at most limit."""
    x = y = 0.0
    n = 0
    while n < limit:
        x2 = x * x
        y2 = y * y
        if x2 + y2 > 4.0:
            return n
        y = (2.0 * x) * y + cy
        x = x2 - y2 + cx
        n += 1
    return limit


def main():
    size = setting("MANDEL_SIZE", 2048)
    frames = setting("MANDEL_FRAMES", 40)
    trace = setting("MANDEL_TRACE", 0)
    width = 3.0
    total = checksum = 0
    for f in range(frames):
        step = width / float(size)
        limit = 256 + 64 * f
        frame = 0
        for py in range(size):
            cy = Y0 + float(py - size // 2) * step
            for px in range(size):
                cx = X0 + float(px - size // 2) * step
                frame += escape(cx, cy, limit)
        if trace:
            print(f"frame {f}: limit {limit}, {frame} iterations")
        total += frame
        checksum += (f + 1) * frame
        width = width * 0.5
    print(f"Iterations: {total}")
    print(f"Checksum: {checksum}")


main()
