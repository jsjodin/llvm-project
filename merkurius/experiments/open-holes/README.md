# Lists built through holes (E5, rebuilt)

The design document's E5 ("E5. Open-term representations (C model)")
modelled in C the code each representation of open terms would produce.
Its program was not kept. This one rebuilds the rows that step 7 of open
types compares with the compiled code (`docs/INPROGRESS.md`, Open types),
from E5's description:

| Row | What it does | Words an element |
| --- | --- | --- |
| `ground-btf` | A ground `list(int)`, built back to front | 2 |
| `ground-ftb` | The same, built front to back by destination passing, as last call modulo constructor does | 2 |
| `hole` | The open layout, built front to back through a hole that each step binds: a variable cell, a box for the int and a cons cell; the sum dereferences each tail and unboxes each int | 4 |
| `hole-trail` | The same, with a trail entry of an address and the old word per binding | 4, and 2 on the trail |

Cells have 3-bit primary tags; a variable is a one-word cell that refers to
itself, with tag 7. Allocation is a bump pointer with a limit check into an
arena that each round resets, as a nursery would be, and whose pages are
touched before the clock starts. The compiled program instead runs on a
heap that the collector copies as it fills.

    clang -O2 -march=native model.c -o model
    model ROW N R

builds and sums [1, ..., N] R times, and prints the row, N, R, the seconds
and the total. `benchmarks/open-holes/compare-c.sh` builds it and runs it
against the compiled programs; the results are in the design document,
"Measurements", "Open terms built through holes".

What it leaves out: E5's other rows (boxed or tagged small ints, a header
bitmap), which decided the representation and need no compiled
counterpart; and the collector, which the arena's reset stands in for.
