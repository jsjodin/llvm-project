// A C model of three heap cell layouts for Merkurius (docs/INPROGRESS.md,
// D1), compiled once per layout with -DLAYOUT_A, -DLAYOUT_B or -DLAYOUT_C:
//
//   A  headerless cells, as Melbourne Mercury's; a cell's layout comes from
//      its type, so copying is type-directed, with forwarding marks in a
//      side bitmap.
//   B  one header word before each cell: size, pointer map, constructor;
//      copying is Cheney's, scanning the to-space linearly.
//   C  pages of cells of one shape (size and pointer map), the shape in a
//      page table; Cheney's scan by shape, forwarding marks in a bitmap.
//
// Every cell pointer has a 3-bit primary tag; constants are small integers.
// Allocation is a bump pointer with a limit check, one per shape under C.
//
// Workloads, each timed and with a checksum:
//   nrev      naive reverse of 400 ints, 3001 rounds, the nursery reset
//             each round (E1's sizes); monomorphic, and polymorphic, where
//             B and C take the header or shape from a type_info at run time
//   list      build a list of N ints, then sum it repeatedly
//   tree      build a complete binary tree node(l, int, r) of depth D, sum it
//   pairs     a list of N pair(int, int), built and summed
//   copy      deep-copy the list and the tree, as send or a minor collection
//             does, into a fresh space
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

typedef uintptr_t W;
#define TAG(p) ((p) & 7)
#define CELL(p) ((W *)((p) & ~(W)7))
#define NIL ((W)0)
#define LEAF ((W)0)

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static W *reserve(size_t bytes) {
  void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  return p;
}

static void die(const char *what) {
  fprintf(stderr, "out of %s\n", what);
  exit(1);
}

// The shapes the workloads use: size in words and pointer map, bit i for
// field i. The cons of a list(int) and of a list(pair) differ in the head.
enum { S_CONS_INT, S_CONS_PTR, S_NODE, S_PAIR, NSHAPES };
static const unsigned shape_size[NSHAPES] = {2, 2, 3, 2};
__attribute__((unused)) static const unsigned shape_ptrs[NSHAPES] = {0x2, 0x3, 0x5, 0x0};
__attribute__((unused)) static const unsigned shape_ctor[NSHAPES] = {1, 1, 1, 0};

// A type_info as polymorphic code has it: whether the type's values are
// pointers, which is all B and C need of it to make a cons.
typedef struct { unsigned is_ptr; } TypeInfo;
static const TypeInfo ti_int = {0};
// Read through memory, as a type_info argument is, so it is not folded.
static const TypeInfo *volatile ti_arg = &ti_int;

//---------------------------------------------------------------------------
// The spaces. A space is a region with a bump pointer (A, B) or, under C,
// pages of 4 KiB, each given to one shape when its shape needs a page.

// 32 KiB blocks, as Immix's.
#define PAGE_WORDS 4096
typedef struct {
  W *base, *end;
#if defined(LAYOUT_C)
  W *next_page;
  W *bump[NSHAPES], *limit[NSHAPES];
  unsigned char *page_shape; // one per page of the space
  W **pages; size_t npages; // pages in the order given out
#else
  W *hp, *hl;
#endif
} Space;

static void space_init(Space *s, size_t words) {
  s->base = reserve(words * sizeof(W));
  s->end = s->base + words;
#if defined(LAYOUT_C)
  s->next_page = s->base;
  for (int i = 0; i < NSHAPES; i++) s->bump[i] = s->limit[i] = NULL;
  s->page_shape = calloc(words / PAGE_WORDS + 1, 1);
  s->pages = malloc((words / PAGE_WORDS + 1) * sizeof(W *));
  s->npages = 0;
#else
  s->hp = s->base;
  s->hl = s->end;
#endif
}

static void space_reset(Space *s) {
#if defined(LAYOUT_C)
  s->next_page = s->base;
  s->npages = 0;
  for (int i = 0; i < NSHAPES; i++) s->bump[i] = s->limit[i] = NULL;
#else
  s->hp = s->base;
#endif
}

static Space nursery, old, to;
static Space *cur = &nursery;

#if defined(LAYOUT_C)
__attribute__((noinline)) static W *new_page(Space *s, int shape) {
  if (s->next_page + PAGE_WORDS > s->end) die("pages");
  W *p = s->next_page;
  s->next_page += PAGE_WORDS;
  s->page_shape[(p - s->base) / PAGE_WORDS] = shape;
  s->pages[s->npages++] = p;
  // Cells do not cross pages: the limit leaves out a partial cell.
  s->bump[shape] = p;
  s->limit[shape] = p + PAGE_WORDS - PAGE_WORDS % shape_size[shape];
  return p;
}
static inline W *alloc_shape(Space *s, int shape) {
  W *p = s->bump[shape];
  W *q = p + shape_size[shape];
  // Before a shape's first page its bump and limit are NULL, so q > limit.
  if (__builtin_expect(q > s->limit[shape], 0)) {
    p = new_page(s, shape);
    q = p + shape_size[shape];
  }
  s->bump[shape] = q;
  return p;
}
#else
__attribute__((noinline)) static void full(void) { die("nursery"); }
static inline W *alloc_words(Space *s, size_t n) {
  W *p = s->hp;
  s->hp = p + n;
  if (__builtin_expect(s->hp > s->hl, 0)) full();
  return p;
}
#endif

#if defined(LAYOUT_B)
// Header: size in bits 0-15, pointer map in 16-47, constructor in 48-62.
// A forwarded cell's header is its new address with bit 63 set, which no
// header has.
#define HDR(size, ptrs, ctor) \
  ((W)(size) | ((W)(ptrs) << 16) | ((W)(ctor) << 48))
#define HDR_SIZE(h) ((h) & 0xffff)
#define HDR_PTRS(h) (((h) >> 16) & 0xffffffff)
#define FORWARDED ((W)1 << 63)
static inline W *alloc_cell(Space *s, W hdr) {
  W *p = alloc_words(s, HDR_SIZE(hdr) + 1);
  p[0] = hdr;
  return p + 1;
}
#endif

// A cell of a shape known when the code is compiled.
static inline W *make(int shape) {
#if defined(LAYOUT_A)
  return alloc_words(cur, shape_size[shape]);
#elif defined(LAYOUT_B)
  return alloc_cell(cur, HDR(shape_size[shape], shape_ptrs[shape],
      shape_ctor[shape]));
#else
  return alloc_shape(cur, shape);
#endif
}

// A cons in polymorphic code, whose element type is the type_info's.
static inline W *make_cons_poly(const TypeInfo *ti) {
#if defined(LAYOUT_A)
  (void) ti;
  return alloc_words(cur, 2);
#elif defined(LAYOUT_B)
  return alloc_cell(cur, HDR(2, 0x2 | ti->is_ptr, 1));
#else
  return alloc_shape(cur, S_CONS_INT + ti->is_ptr);
#endif
}

static inline W cons(W h, W t) {
  W *c = make(S_CONS_INT);
  c[0] = h;
  c[1] = t;
  return (W) c | 1;
}

static inline W cons_poly(const TypeInfo *ti, W h, W t) {
  W *c = make_cons_poly(ti);
  c[0] = h;
  c[1] = t;
  return (W) c | 1;
}

//---------------------------------------------------------------------------
// nrev, as E1's: append(nrev(T), [H]).

static W append(W xs, W ys) {
  if (xs == NIL) return ys;
  W *c = CELL(xs);
  return cons(c[0], append(c[1], ys));
}
static W nrev(W xs) {
  if (xs == NIL) return NIL;
  W *c = CELL(xs);
  return append(nrev(c[1]), cons(c[0], NIL));
}
static W append_poly(const TypeInfo *ti, W xs, W ys) {
  if (xs == NIL) return ys;
  W *c = CELL(xs);
  return cons_poly(ti, c[0], append_poly(ti, c[1], ys));
}
static W nrev_poly(const TypeInfo *ti, W xs) {
  if (xs == NIL) return NIL;
  W *c = CELL(xs);
  return append_poly(ti, nrev_poly(ti, c[1]), cons_poly(ti, c[0], NIL));
}

static void bench_nrev(int poly) {
  cur = &old;
  W list = NIL;
  for (int i = 400; i >= 1; i--) list = cons(i, list);
  cur = &nursery;
  W head = 0;
  double t0 = now();
  for (int round = 0; round < 3001; round++) {
    space_reset(&nursery);
    W r = poly ? nrev_poly(ti_arg, list) : nrev(list);
    head += CELL(r)[0];
  }
  double t = now() - t0;
  printf("nrev%s\t400x3001\t%.3f s\t%.2f ns/cell\tsum %lu\n",
      poly ? "-poly" : "", t, 1e9 * t / (3001.0 * 80600), (unsigned long) head);
}

//---------------------------------------------------------------------------
// Lists, trees and pairs.

static W build_list(long n) {
  W l = NIL;
  for (long i = n; i >= 1; i--) l = cons(i, l);
  return l;
}
static W sum_list(W l) {
  W s = 0;
  while (l != NIL) {
    W *c = CELL(l);
    s += c[0];
    l = c[1];
  }
  return s;
}

static W build_tree(int depth, W *counter) {
  if (depth == 0) return LEAF;
  W l = build_tree(depth - 1, counter);
  W v = (*counter)++;
  W r = build_tree(depth - 1, counter);
  W *c = make(S_NODE);
  c[0] = l;
  c[1] = v;
  c[2] = r;
  return (W) c | 1;
}
static W sum_tree(W t) {
  W s = 0;
  while (t != LEAF) {
    W *c = CELL(t);
    s += c[1] + sum_tree(c[0]);
    t = c[2];
  }
  return s;
}

static W build_pairs(long n) {
  W l = NIL;
  for (long i = n; i >= 1; i--) {
    W *p = make(S_PAIR);
    p[0] = i;
    p[1] = 2 * i;
    W *c = make(S_CONS_PTR);
    c[0] = (W) p | 1;
    c[1] = l;
    l = (W) c | 1;
  }
  return l;
}
static W sum_pairs(W l) {
  W s = 0;
  while (l != NIL) {
    W *c = CELL(l);
    W *p = CELL(c[0]);
    s += p[0] + p[1];
    l = c[1];
  }
  return s;
}

//---------------------------------------------------------------------------
// Deep copy into `to`, as send copies a message into a fragment and a minor
// collection copies the survivors, preserving sharing.

#if defined(LAYOUT_A) || defined(LAYOUT_C)
// Forwarding marks, a bit per word of the space copied from: a marked
// cell's first word is its new address.
static uint64_t *marks;
static W *marks_base;
static inline int marked(W *c) {
  size_t i = c - marks_base;
  return (marks[i >> 6] >> (i & 63)) & 1;
}
static inline void mark(W *c) {
  size_t i = c - marks_base;
  marks[i >> 6] |= (uint64_t) 1 << (i & 63);
}
// Cleared over the part of the space in use, as a collector clears them.
static void marks_init(Space *from, size_t used) {
  size_t words = from->end - from->base;
  if (!marks) marks = calloc(words / 64 + 1, 8);
  else memset(marks, 0, (used / 64 + 1) * 8);
  marks_base = from->base;
}
#endif

#if defined(LAYOUT_A)
// Type-directed: the copier knows each field's type. A type descriptor
// here gives, per constructor of the type, the size and each field's type:
// raw, or a pointer to a value of a type, by descriptor.
typedef struct Desc {
  unsigned size;
  const struct Desc *field[3]; // NULL: raw
} Desc;
static const Desc d_list_int, d_tree, d_list_pair, d_pair;
static const Desc d_list_int = {2, {NULL, &d_list_int}};
static const Desc d_tree = {3, {&d_tree, NULL, &d_tree}};
static const Desc d_pair = {2, {NULL, NULL}};
static const Desc d_list_pair = {2, {&d_pair, &d_list_pair}};

// The work: fields of copied cells still to copy, each with its type; the
// cells have no headers, so the to-space cannot be scanned linearly.
typedef struct { W *slot; const Desc *d; } Work;
static Work *work;
static size_t work_cap;

static inline W forward(W v, const Desc *d, size_t *top) {
  if (TAG(v) == 0) return v; // a constant
  W *c = CELL(v);
  if (marked(c)) return c[0] | TAG(v);
  W *n = alloc_words(&to, d->size);
  memcpy(n, c, d->size * sizeof(W));
  c[0] = (W) n;
  mark(c);
  for (unsigned i = 0; i < d->size; i++)
    if (d->field[i]) {
      if (*top == work_cap) die("work stack");
      work[(*top)++] = (Work){&n[i], d->field[i]};
    }
  return (W) n | TAG(v);
}
static W deep_copy(W root, const Desc *d) {
  if (!work) {
    work_cap = (size_t) 1 << 26;
    work = malloc(work_cap * sizeof(Work));
  }
  size_t top = 0;
  W r = forward(root, d, &top);
  while (top) {
    Work w = work[--top];
    *w.slot = forward(*w.slot, w.d, &top);
  }
  return r;
}
#define COPY(root, desc) deep_copy(root, &desc)

#elif defined(LAYOUT_B)
// Cheney: the to-space is scanned linearly, each cell's header giving its
// size and pointer fields.
static inline W forward(W v) {
  if (TAG(v) == 0) return v;
  W *c = CELL(v);
  W h = c[-1];
  if (h & FORWARDED) return (h & ~FORWARDED) | TAG(v);
  W *n = alloc_cell(&to, h);
  memcpy(n, c, HDR_SIZE(h) * sizeof(W));
  c[-1] = (W) n | FORWARDED;
  return (W) n | TAG(v);
}
static W deep_copy(W root) {
  W *scan = to.hp;
  W r = forward(root);
  while (scan < to.hp) {
    W h = scan[0];
    W *f = scan + 1;
    unsigned ptrs = HDR_PTRS(h);
    for (unsigned i = 0; ptrs; i++, ptrs >>= 1)
      if (ptrs & 1) f[i] = forward(f[i]);
    scan = f + HDR_SIZE(h);
  }
  return r;
}
#define COPY(root, desc) deep_copy(root)

#else
// Cheney by shape: each to-space page is scanned with its shape's size and
// pointer map, pages in the order given out, until no page is unscanned.
static Space *from_space;
static inline W forward(W v) {
  if (TAG(v) == 0) return v;
  W *c = CELL(v);
  if (marked(c)) return c[0] | TAG(v);
  int shape = from_space->page_shape[(c - from_space->base) / PAGE_WORDS];
  W *n = alloc_shape(&to, shape);
  memcpy(n, c, shape_size[shape] * sizeof(W));
  c[0] = (W) n;
  mark(c);
  return (W) n | TAG(v);
}
static inline W *page_start(Space *sp, W *p) {
  return sp->base + (size_t) (p - sp->base) / PAGE_WORDS * PAGE_WORDS;
}
static inline W *page_full(W *page, int shape) {
  return page + PAGE_WORDS - PAGE_WORDS % shape_size[shape];
}
// Scans from `scan` to where its page's cells end: the page's last whole
// cell, or the shape's bump pointer when this is the page being filled,
// read again after each cell, since forwarding adds cells.
static W *scan_page(W *scan, int shape) {
  W *page = page_start(&to, scan);
  for (;;) {
    W *b = to.bump[shape];
    W *limit = b > page && page_start(&to, b - 1) == page ? b
                                                         : page_full(page, shape);
    if (scan >= limit) return scan;
    unsigned ptrs = shape_ptrs[shape];
    for (unsigned i = 0; ptrs; i++, ptrs >>= 1)
      if (ptrs & 1) scan[i] = forward(scan[i]);
    scan += shape_size[shape];
  }
}
static W deep_copy(W root) {
  size_t page = to.npages;
  // Where the scan stopped in each shape's page still being filled.
  W *pending[NSHAPES] = {0};
  W r = forward(root);
  for (int progress = 1; progress;) {
    progress = 0;
    for (int s = 0; s < NSHAPES; s++)
      if (pending[s]) {
        W *e = scan_page(pending[s], s);
        progress |= e != pending[s];
        pending[s] = e < page_full(page_start(&to, pending[s]), s) ? e : NULL;
      }
    while (page < to.npages) {
      W *p = to.pages[page++];
      int s = to.page_shape[(p - to.base) / PAGE_WORDS];
      if (pending[s]) { // its earlier page is full now
        scan_page(pending[s], s);
        pending[s] = NULL;
      }
      W *e = scan_page(p, s);
      if (e < page_full(p, s)) pending[s] = e;
      progress = 1;
    }
  }
  return r;
}
#define COPY(root, desc) deep_copy(root)
#endif

static long words_used(Space *s) {
#if defined(LAYOUT_C)
  // Whole pages, an upper bound: a page's tail that no cell fits is lost.
  return (long) s->npages * PAGE_WORDS;
#else
  return s->hp - s->base;
#endif
}

static W copy_of(W root, int is_tree) {
  space_reset(&to);
#if defined(LAYOUT_A)
  marks_init(cur, words_used(cur));
  return is_tree ? COPY(root, d_tree) : COPY(root, d_list_int);
#elif defined(LAYOUT_C)
  marks_init(cur, words_used(cur));
  from_space = cur;
  (void) is_tree;
  return COPY(root, 0);
#else
  (void) is_tree;
  return COPY(root, 0);
#endif
}

//---------------------------------------------------------------------------

// Times `reps` builds of n elements, then sums the last one repeatedly,
// then copies it (rebuilt before each copy) and sums the copy. Prints a row
// of the per-element costs.
typedef W (*Build)(long);
typedef W (*Sum)(W);
static W build_tree_n(long n) {
  int depth = 1;
  while (((long) 1 << depth) - 1 < n) depth++;
  W counter = 0;
  return build_tree(depth, &counter);
}
static void bench(const char *name, Build build, Sum sum, long n, int copies,
    int is_tree) {
  if (is_tree) {
    int depth = 1;
    while (((long) 1 << depth) - 1 < n) depth++;
    n = ((long) 1 << depth) - 1;
  }
  long reps = ((long) 1 << 24) / n;
  if (reps < 1) reps = 1;
  W l = NIL;
  double t0 = now();
  for (long r = 0; r < reps; r++) {
    space_reset(&nursery);
    l = build(n);
    __asm__ volatile("" : "+r"(l));
  }
  double tb = (now() - t0) / reps;
  double bytes = 8.0 * words_used(&nursery) / n;
  long passes = ((long) 1 << 27) / n;
  if (passes < 3) passes = 3;
  W s = 0;
  t0 = now();
  for (long i = 0; i < passes; i++) {
    __asm__ volatile("" : "+r"(l));
    s += sum(l);
  }
  double ts = (now() - t0) / passes;
  double tc = 0, tsc = 0;
  W sc = 0;
  if (copies) {
    long creps = ((long) 1 << 22) / n;
    if (creps < 1) creps = 1;
    W c = NIL;
    for (long r = 0; r < creps; r++) {
      space_reset(&nursery);
      l = build(n);
      t0 = now();
      c = copy_of(l, is_tree);
      tc += now() - t0;
    }
    tc /= creps;
    t0 = now();
    for (long i = 0; i < passes; i++) {
      __asm__ volatile("" : "+r"(c));
      sc += sum(c);
    }
    tsc = (now() - t0) / passes;
    sc /= passes;
  }
  printf("%s\t%ld\t%.1f\t%.3f\t%.3f\t%.3f\t%.3f\t%lu\t%lu\n", name, n, bytes,
      1e9 * tb / n, 1e9 * ts / n, 1e9 * tc / n, 1e9 * tsc / n,
      (unsigned long) (s / passes), (unsigned long) sc);
  fflush(stdout);
}

int main(int argc, char **argv) {
  const char *what = argc > 1 ? argv[1] : "nrev";
  long n = argc > 2 ? atol(argv[2]) : 1000000;
  space_init(&nursery, (size_t) 1 << 31); // 16 GiB reserved, touched lazily
  space_init(&old, (size_t) 1 << 20);
  space_init(&to, (size_t) 1 << 31);
  // Rows: workload, elements, bytes/elem, build, sum, copy, sum of the
  // copy (ns/elem), checksums.
  if (!strcmp(what, "nrev")) bench_nrev(0);
  else if (!strcmp(what, "nrev-poly")) bench_nrev(1);
  else if (!strcmp(what, "list")) bench("list", build_list, sum_list, n, 1, 0);
  else if (!strcmp(what, "tree")) bench("tree", build_tree_n, sum_tree, n, 1, 1);
  else if (!strcmp(what, "pairs")) bench("pairs", build_pairs, sum_pairs, n, 0, 0);
  return 0;
}
