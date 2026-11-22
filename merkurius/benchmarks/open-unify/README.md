# Unification of open lists

Two lists of the ints 1 to N are built through holes, one with a new hole
for every 100th element, then unified, which binds those holes to the other
list's elements, and the first is summed, R times. Results are in the design
document, "Measurements", "Unification of open terms".

The unification is `X = Y` on `open(list(int))`, which calls the type's own
unify procedure (docs/INPROGRESS.md, Open types, step 4(d) part 3): each
element is dereferenced and compared inline, and the tail is a tail call.
Before it, the runtime's generic walker took the type_info, and past 16384
cells remembered every pair of cells in a table, against cyclic terms.

    run.sh [build-dir] [N] [R]

or by hand:

    merkc -O2 -fopen-types open-unify.m -o open-unify
    ./open-unify 100000 100

The heap collects as it fills, moving the open terms and the trail's
entries (docs/INPROGRESS.md, Open types, step 5). Measurements before step
5 gave the run one heap chunk instead, whose page faults were in the
elapsed time.
