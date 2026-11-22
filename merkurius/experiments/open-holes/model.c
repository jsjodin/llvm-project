// E5's C model of lists built through holes, rebuilt (README.md): the code
// the open layout would produce, against ground lists, with 3-bit primary
// tags and a bump allocator with a limit check into an arena that each
// round resets, as a nursery would be.
//
//   model ROW N R
//
// builds a list of the ints 1 to N and sums it, R times, and prints the
// row, N, R, the seconds and the total. ROW is one of:
//
//   ground-btf    ground list(int), built back to front
//   ground-ftb    ground list(int), built front to back by destination
//                 passing, as last call modulo constructor does
//   hole          the open layout, built front to back through a hole that
//                 each step binds: a variable cell, a box for the int and
//                 a cons cell an element; the sum dereferences each tail
//   hole-trail    the same, with a trail entry {address, old} per binding
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint64_t Word;

enum { kVariableTag = 7, kConsTag = 1 };
#define TAG(w) ((w) & 7)
#define CELL(w) ((Word *)((w) & ~(Word)7))

static Word *heap, *next, *limit;

static void Exhausted(void) {
  fputs("model: arena exhausted\n", stderr);
  exit(1);
}

// The bump allocation with its limit check, as compiled code has it.
static inline Word *Allocate(size_t words) {
  Word *cell = next;
  next += words;
  if (__builtin_expect(next > limit, 0)) {
    Exhausted();
  }
  return cell;
}

static inline Word Cons(Word head, Word tail) {
  Word *cell = Allocate(2);
  cell[0] = head;
  cell[1] = tail;
  return (Word)cell | kConsTag;
}

// [] is the word 0, as a constant of tag 0.
static Word GroundBackToFront(int64_t n) {
  Word list = 0;
  for (int64_t i = n; i >= 1; --i) {
    list = Cons((Word)i, list);
  }
  return list;
}

static Word GroundFrontToBack(int64_t n) {
  Word list = 0;
  Word *hole = &list;
  for (int64_t i = 1; i <= n; ++i) {
    Word *cell = Allocate(2);
    cell[0] = (Word)i;
    *hole = (Word)cell | kConsTag;
    hole = &cell[1];
  }
  *hole = 0;
  return list;
}

static int64_t SumGround(Word list) {
  int64_t sum = 0;
  while (list != 0) {
    Word *cell = CELL(list);
    sum += (int64_t)cell[0];
    list = cell[1];
  }
  return sum;
}

static inline Word NewVariable(void) {
  Word *cell = Allocate(1);
  *cell = (Word)cell | kVariableTag;
  return *cell;
}

static inline Word Box(int64_t i) {
  Word *box = Allocate(1);
  *box = (Word)i;
  return (Word)box;
}

typedef struct {
  Word *address;
  Word old;
} TrailEntry;

static TrailEntry *trail, *trailTop, *trailLimit;

// Binds an unbound variable, trailing the binding if `trailed`.
static inline void Bind(Word variable, Word value, int trailed) {
  Word *cell = CELL(variable);
  if (trailed) {
    if (__builtin_expect(trailTop == trailLimit, 0)) {
      Exhausted();
    }
    trailTop->address = cell;
    trailTop->old = *cell;
    ++trailTop;
  }
  *cell = value;
}

static inline Word Deref(Word w) {
  while (TAG(w) == kVariableTag) {
    Word next = *CELL(w);
    if (next == w) {
      break;
    }
    w = next;
  }
  return w;
}

// L = [I | T], T a new variable, until I > N, which binds L to [].
static Word ThroughHoles(int64_t n, int trailed) {
  Word list = NewVariable();
  Word hole = list;
  for (int64_t i = 1; i <= n; ++i) {
    Word tail = NewVariable();
    Bind(hole, Cons(Box(i), tail), trailed);
    hole = tail;
  }
  Bind(hole, 0, trailed);
  return list;
}

static int64_t SumOpen(Word list) {
  int64_t sum = 0;
  for (;;) {
    list = Deref(list);
    if (list == 0) {
      return sum;
    }
    Word *cell = CELL(list);
    sum += *(int64_t *)Deref(cell[0]);
    list = cell[1];
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fputs("usage: model ground-btf|ground-ftb|hole|hole-trail N R\n", stderr);
    return 2;
  }
  const char *row = argv[1];
  int64_t n = atoll(argv[2]), rounds = atoll(argv[3]);
  int kind = !strcmp(row, "ground-btf") ? 0
      : !strcmp(row, "ground-ftb")      ? 1
      : !strcmp(row, "hole")            ? 2
      : !strcmp(row, "hole-trail")      ? 3
                                        : -1;
  if (kind < 0 || n < 1 || rounds < 1) {
    fputs("model: bad arguments\n", stderr);
    return 2;
  }
  size_t words = 4 * (size_t)n + 16;
  heap = malloc(words * sizeof(Word));
  trail = malloc(((size_t)n + 1) * sizeof(TrailEntry));
  if (!heap || !trail) {
    Exhausted();
  }
  // Touch the arena once, so that its page faults are not in the time.
  memset(heap, 0, words * sizeof(Word));
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  int64_t total = 0;
  for (int64_t r = 0; r < rounds; ++r) {
    next = heap;
    limit = heap + words;
    trailTop = trail;
    trailLimit = trail + n + 1;
    switch (kind) {
    case 0:
      total += SumGround(GroundBackToFront(n));
      break;
    case 1:
      total += SumGround(GroundFrontToBack(n));
      break;
    default:
      total += SumOpen(ThroughHoles(n, kind == 3));
      break;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  double seconds = (double)(end.tv_sec - start.tv_sec) +
      1e-9 * (double)(end.tv_nsec - start.tv_nsec);
  printf("%s\t%lld\t%lld\t%.3f\t%lld\n", row, (long long)n,
      (long long)rounds, seconds, (long long)total);
  return 0;
}
