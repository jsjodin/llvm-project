# Open terms built through holes

A list of N ints built front to back through a hole that each step binds,
`L = [I | T]` with T a new variable, then summed by reading it back as an
open term, R times: the shape of E5's last rows (design document, "E5.
Open-term representations"), which a C model measured at 2.3 times the
ground front-to-back build. Results are in the design document,
"Measurements", "Open terms built through holes".

Each step allocates a variable cell, a box for the int and a cell, binds
the hole, and trails the binding, because the round's if-then-else
condition is live; the sum dereferences each tail and unboxes each int.

    run.sh [build-dir] [N] [R]

or by hand:

    merkc -O2 -fopen-types open-holes.m -o open-holes
    ./open-holes 100000 100

The heap collects as it fills, moving the open terms and the trail's
entries (design document, "Open terms as built"). Measurements before step
5 gave the run one heap chunk instead, whose page faults were in the
elapsed time.

`compare-c.sh` runs the program against E5's C model, rebuilt in
`experiments/open-holes`, and against `ground-holes.m`, the same list as a
ground `list(int)` built front to back by last call modulo constructor, at
E5's size, 1,000,000 elements 200 times, on one pinned core:

    compare-c.sh [build-dir] [N] [R] [core]
