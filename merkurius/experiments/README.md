# Experiments

Models and measurements that a design decision rests on, kept so that they
can be run again on another machine or after a change. Each directory has a
script that builds and runs it, and a README with what it measures, the
results it gave and on what, and what it leaves out. The decisions they
inform are in `docs/INPROGRESS.md` while open, and in the design document
once made. The design document's Experiments section, E1 to E5, predates
this directory; their programs were not kept. E5's rows that compiled code
is compared with are rebuilt in `open-holes`.

| Experiment | Decision | What it measures |
| --- | --- | --- |
| [`layout`](layout/README.md) | D1, heap cell layout | Headerless cells (A), a header word (B) and pages of one shape (C): allocation, traversal across cache levels, collector copies, polymorphic allocation |
| [`open-holes`](open-holes/README.md) | E5, rebuilt for open types step 7 | A list built through holes in the open layout, with and without a trail, against ground lists built back to front and front to back |
