# Heap cell layouts (D1)

A C model of the three cell layouts of `docs/INPROGRESS.md`, decision D1,
measured on the same workloads, so that the choice rests on numbers,
including what each layout's size does to cache use.

| Layout | Cell | How the collector and a copy find pointers |
| --- | --- | --- |
| A | Headerless, as Melbourne Mercury's: a cons is 2 words | The cell's type: a type-directed copy with a work stack, forwarding marks in a side bitmap |
| B | One header word before the cell: size, pointer map, constructor | The header: Cheney's linear scan of the to-space |
| C | Headerless, in 32 KiB pages of cells of one shape, the shape in a page table | The page's shape: Cheney's scan page by page, marks in a bitmap |

Every layout tags pointers with a 3-bit primary tag and allocates with a bump
pointer and a limit check; C has one bump pointer per shape.

## Running

```sh
merkurius/experiments/layout/run.sh [cc] [core]
```

It builds `model.c` once per layout, runs every workload at sizes from L1 to
DRAM pinned to one core, five times, and writes each run to
`out/results.tsv` and the medians to `out/summary.tsv`. `summarize.py` also
checks that every layout and run computed the same checksums. One binary
can be run by hand, `out/modelB tree 2000000`, which prints one row.

## Workloads

- **nrev:** naive reverse of 400 ints, 3001 rounds, about 242 million cons
  cells, the nursery emptied each round (E1's sizes). Monomorphic, and
  polymorphic, where B takes the head's pointer bit from a `type_info` and C
  the shape, at run time, as polymorphic code such as `append` must: a
  cons's pointer map depends on its element type.
- **list:** build a list of N ints, sum it repeatedly, deep-copy it, sum the
  copy.
- **tree:** the same for a complete binary tree `node(tree, int, tree)`.
- **pairs:** a list of N `pair(int, int)`, built and summed: two cells per
  element.

The copy is a collector's: it overwrites the old cells with forwarding
marks. `send` cannot, since the sender keeps its message, so messages need a
non-destructive copy whatever the layout.

## Results

AMD Ryzen Threadripper PRO 5965WX (32 KiB L1d and 512 KiB L2 per core,
32 MiB L3 per 8-core complex), one core, clang 24 (the build's) at `-O2
-march=native`, 2026-11-04. Medians of 5 runs, nanoseconds per element.

**Size, bytes per element** (C's at 1000 elements includes the unused rest
of its pages):

| | A | B | C |
| --- | --- | --- | --- |
| list | 16 | 24 | 16 |
| tree | 24 | 32 | 24 |
| pairs | 32 | 48 | 32 |

**nrev**, ns per cons cell:

| | A | B | C |
| --- | --- | --- | --- |
| monomorphic | 1.95 | 2.17 (+11%) | 2.03 (+4%) |
| polymorphic | 1.97 | 2.31 (+17%) | 2.68 (+36%) |

**Traversal (sum)**, ns per element, as the working set leaves each cache:

| Workload | Elements | A | B | C |
| --- | --- | --- | --- | --- |
| list | 256K (L3) | 1.22 | 1.21 | 1.22 |
| list | 2M | 1.23 | 1.40 (+14%) | 1.23 |
| list | 16M (DRAM) | 1.25 | 1.45 (+16%) | 1.26 |
| tree | 16K (L2) | 1.45 | 1.70 (+18%) | 1.58 |
| tree | 256K | 1.60 | 1.93 (+21%) | 1.73 |
| tree | 2M | 3.54 | 5.50 (+55%) | 3.66 |
| tree | 16M | 5.04 | 7.53 (+50%) | 5.23 |
| pairs | 2M | 1.76 | 2.53 (+44%) | 1.39 (−21%) |
| pairs | 16M | 1.82 | 2.64 (+45%) | 1.42 (−22%) |

**Build**, ns per element: B's extra word costs 30–50% once the cells leave
L2 (list at 16M: A 5.7, B 8.5, C 6.1; tree at 16M: A 11.7, B 14.6, C 11.1);
within the caches B is 3–40% slower and C up to 2.6× slower for lists in
this model, its per-shape bump pointer and page changes not tuned.

**Collector copy**, ns per cell: list A 4.9–10.8, B 8.0–16.9, C 9.3–15.2;
tree A 6.5–16.2, B 5.0–23.7, C 6.3–20.8, the first number at 1000 elements,
the second at 16M. B's Cheney scan is the fastest for small trees, A's
typed copy for lists and for anything large.

## What the numbers say

- **B's header costs what its size predicts once data leaves the cache:**
  +14–16% on lists, +50–55% on trees, +45% on pairs, +30–50% on building.
  Inside the caches it costs 0–21%. nrev, E1's allocation benchmark, whose
  nursery round is 1.3 MB under A and 1.9 MB under B, costs +11%, and +17%
  in polymorphic code, where the header's pointer bit comes from the
  `type_info`.
- **C keeps A's size** and its segregation can beat A: the spine of a list
  of pairs is dense, 4 cons cells to a cache line instead of 2, so summing
  it is 21% faster than A's. But polymorphic allocation must pick the bump
  pointer by a shape known only at run time (+36% on nrev), and a heap
  needs a page for each shape it uses: 32.8 bytes per list element at 1000
  elements against 16. With a heap per process, a small process would pay
  a page per shape, which is the wrong cost for many small processes.
- **A is the smallest and fastest** on every workload but small-tree
  copies, and needs nothing at run time from polymorphic code. Its cost is
  in the compiler and collector, not in the numbers: every root's
  `type_info` must be findable from the stack maps, so the compiler keeps
  each polymorphic value's `type_info` live as long as the value; the
  to-space cannot be scanned linearly, so the copy needs a typed work
  stack; and the heap is not self-describing for debuggers and heap
  profilers.
- **The order a copy leaves cells in is the copier's, not the layout's.**
  After B's and C's breadth-first Cheney copy, summing the 16M-node tree
  takes 1.55–1.59 ns, against 6.09 after A's depth-first copy with its work
  stack, which visits the right child first. A typed copier can copy in
  either order, so this is a choice for the collector, not for D1.

## What the model leaves out

Real roots and stack maps, write barriers, LLVM's code for our own `mrt`
operations rather than C's, a tuned allocator for C, open terms' variable
cells, and a non-destructive copy for `send`. It measures one core, so
nothing about contention between processes' heaps.
