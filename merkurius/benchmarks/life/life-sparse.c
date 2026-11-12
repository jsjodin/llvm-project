// Conway's Game of Life on sparse tiles in C with OpenMP, the bar for
// life-sparse.m (docs/INPROGRESS.md, sparse tiles on one place, step s6):
// the same world, tiles, step and rules, as a C programmer would write them
// for a GPU. See README.md.
//
// The host keeps the tiles' tables and does the bookkeeping; the GPU keeps
// the cells. Each step is one kernel, a team for each active tile and its
// threads over the tile's cells, which reduce the tile's live cells and live
// edges. The host copies those back, a pair for each active tile, decides
// which tiles end and start, and only if some do copies the active tiles and
// their neighbours to the GPU and zeroes the new tiles' cells there, a
// second kernel. life-sparse.m does all of it in one GPU process, which
// starts no kernel and copies nothing.
//
// - The world is LIFE_WORLD cells square (8192), dead beyond its edge, in
//   tiles of LIFE_TILE cells square (64), a power of two that divides it,
//   with a soup of side LIFE_SOUP (256) at its centre, as life.c's
//   LIFE_SOUP, which gives the same population and checksum for the same
//   world and LIFE_STEPS (1000).
// - The pool: two arrays of LIFE_CAPACITY tiles' cells, uint8_t, on the GPU,
//   which swap after each step. Tile 0 is the dead tile, all zero.
// - Nine neighbours a tile, by d = (dy + 1) * 3 + dx + 1, the middle one the
//   tile itself, so that the cell at row r and column c of a tile, each from
//   -1 to T, is in neighbour ((r >> log) + 1) * 3 + (c >> log) + 1.
// - A tile with no live cells, and no live edge of a neighbour facing it,
//   for QUIET_STEPS steps ends; each live edge or corner facing no tile,
//   inside the world, starts one there.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// The kernels are separate functions, as in life-sparse.m.
#define KERNEL __attribute__((noinline)) static

enum { QUIET_STEPS = 4, ALL_EDGES = 0x1ef, ENDS = 0x200 };

static long setting(const char *name, long fallback) {
  const char *s = getenv(name);
  return s && *s ? strtol(s, NULL, 10) : fallback;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

// Whether the cell of index k in the world starts alive, if it is in the
// soup, as in life.c.
#pragma omp declare target
static inline uint8_t alive(int64_t k) {
  int64_t h = (k * 2654435761) & 4294967295;
  return (h >> 16) % 100 < 30 ? 1 : 0;
}

// The directions that a live cell at row r and column c of a tile whose
// last row and column are m makes live: its edge, if any, each, and the
// corner.
static inline int edge_bits(int64_t r, int64_t c, int64_t m) {
  int dy = r == 0 ? -1 : r == m ? 1 : 0;
  int dx = c == 0 ? -1 : c == m ? 1 : 0;
  int bits = 0;
  if (dy != 0)
    bits |= 1 << ((dy + 1) * 3 + 1);
  if (dx != 0)
    bits |= 1 << (3 + dx + 1);
  if (dy != 0 && dx != 0)
    bits |= 1 << ((dy + 1) * 3 + dx + 1);
  return bits;
}
#pragma omp end declare target

// The world's shape and the host's tables of the tiles.
struct world {
  int64_t tile, log, side, size, capacity;
  int32_t *dir; // A tile for each place, or 0.
  int32_t *pos; // Each tile's place.
  int32_t *nb; // Each tile's nine neighbours, by tile.
  int32_t *live; // Each tile's live cells after the last step.
  uint16_t *edges; // Each tile's live edges after the last step.
  uint8_t *quiet; // The steps each tile has been quiet.
  int32_t *actives; // The active tiles, nactive of them.
  int32_t *free; // The free tiles, nfree of them.
  uint16_t *fate; // Each active tile's fate, by its slot.
  int64_t nactive, nfree;
  int overflow;
  // What the GPU reads, by the active tiles' slots: their neighbours, and
  // the step's results.
  int32_t *slot_nb;
  int32_t *slot_live;
  uint16_t *slot_edges;
};

// The place in direction d of place p, or -1 if it is outside the world.
static int64_t beside(const struct world *w, int64_t p, int d) {
  int64_t y = p / w->side + d / 3 - 1, x = p % w->side + d % 3 - 1;
  if (y < 0 || y >= w->side || x < 0 || x >= w->side)
    return -1;
  return y * w->side + x;
}

// A tile at the place, from the free list, linked to its neighbours both
// ways and active; or the overflow set if none is free.
static void start(struct world *w, int64_t place) {
  if (w->nfree == 0) {
    w->overflow = 1;
    return;
  }
  int32_t i = w->free[--w->nfree];
  w->dir[place] = i;
  w->pos[i] = (int32_t)place;
  w->quiet[i] = 0;
  for (int d = 0; d < 9; d++) {
    int64_t there = beside(w, place, d);
    int32_t n = d == 4 ? i : there < 0 ? 0 : w->dir[there];
    w->nb[i * 9 + d] = n;
    if (d != 4 && n != 0)
      w->nb[n * 9 + 8 - d] = i;
  }
  w->actives[w->nactive++] = i;
}

// Each active tile's fate after a step: its quiet steps, which end it at
// QUIET_STEPS, and the directions where it starts a tile. Whether any tile
// has one.
static int decide(struct world *w) {
  int pending = 0;
  for (int64_t j = 0; j < w->nactive; j++) {
    int32_t i = w->actives[j];
    const int32_t *n = w->nb + i * 9;
    int missing = 0, facing = 0;
    for (int d = 0; d < 9; d++) {
      if (d == 4)
        continue;
      if (n[d] == 0)
        missing |= 1 << d;
      else
        facing |= w->edges[n[d]] & (1 << (8 - d));
    }
    int inside = 0;
    for (int d = 0; d < 9; d++)
      if (beside(w, w->pos[i], d) >= 0)
        inside |= 1 << d;
    int fate = w->edges[i] & missing & inside;
    int q = w->live[i] == 0 && facing == 0 ? w->quiet[i] + 1 : 0;
    w->quiet[i] = (uint8_t)q;
    if (q >= QUIET_STEPS)
      fate |= ENDS;
    w->fate[j] = (uint16_t)fate;
    pending |= fate != 0;
  }
  return pending;
}

// The tiles whose fates end them end, each slot taken by the last one's.
static void finish(struct world *w) {
  for (int64_t j = 0; j < w->nactive;) {
    if (!(w->fate[j] & ENDS)) {
      j++;
      continue;
    }
    int32_t i = w->actives[j];
    for (int d = 0; d < 9; d++) {
      int32_t n = w->nb[i * 9 + d];
      if (d != 4 && n != 0)
        w->nb[n * 9 + 8 - d] = 0;
    }
    w->dir[w->pos[i]] = 0;
    w->free[w->nfree++] = i;
    w->nactive--;
    w->actives[j] = w->actives[w->nactive];
    w->fate[j] = w->fate[w->nactive];
  }
}

// The tiles the fates of the first kept active tiles ask for, where none
// has started since.
static void grow(struct world *w, int64_t kept) {
  for (int64_t j = 0; j < kept; j++) {
    int bits = w->fate[j] & 0x1ff;
    int32_t i = w->actives[j];
    for (int d = 0; d < 9 && bits != 0; d++) {
      if (!(bits & (1 << d)) || w->nb[i * 9 + d] != 0)
        continue;
      int64_t place = beside(w, w->pos[i], d);
      if (place >= 0)
        start(w, place);
    }
  }
}

// The active tiles' neighbours by their slots, for the GPU.
static void slots(struct world *w) {
  for (int64_t j = 0; j < w->nactive; j++)
    memcpy(w->slot_nb + j * 9, w->nb + w->actives[j] * 9, 9 * sizeof(int32_t));
}

// The soup's cells in the active tiles, which are the soup's.
KERNEL void init(
    const struct world *w, int64_t from, int64_t soup, uint8_t *cur) {
  int64_t t = w->tile, log = w->log, side = w->side, size = w->size;
  int64_t nactive = w->nactive, capacity = w->capacity;
  int64_t cells = capacity * t * t;
  const int32_t *actives = w->actives, *pos = w->pos;
#pragma omp target teams distribute map(to : actives[0 : nactive], \
        pos[0 : capacity]) map(tofrom : cur[0 : cells])
  for (int64_t j = 0; j < nactive; j++) {
    int64_t i = actives[j], p = pos[i];
    int64_t top = (p / side) << log, left = (p % side) << log;
#pragma omp parallel for
    for (int64_t k = 0; k < t * t; k++) {
      int64_t row = top + (k >> log), column = left + (k & (t - 1));
      cur[(i << (2 * log)) + k] = row >= from && row < from + soup &&
              column >= from && column < from + soup
          ? alive(row * size + column)
          : 0;
    }
  }
}

// One step: next's cells of the active tiles from cur's, and each one's
// live cells and live edges, by its slot.
KERNEL void step(int64_t log, int64_t nactive, int64_t cells,
    const int32_t *actives, const int32_t *slot_nb, const uint8_t *cur,
    uint8_t *next, int32_t *slot_live, uint16_t *slot_edges) {
  int64_t t = (int64_t)1 << log, m = t - 1;
#pragma omp target teams distribute map( \
        to : actives[0 : nactive], slot_nb[0 : nactive * 9], cur[0 : cells]) \
    map(tofrom : next[0 : cells], slot_live[0 : nactive], \
            slot_edges[0 : nactive])
  for (int64_t j = 0; j < nactive; j++) {
    int64_t base = (int64_t)actives[j] << (2 * log);
    const int32_t *n = slot_nb + j * 9;
    int32_t live = 0;
    int edges = 0;
#pragma omp parallel for reduction(+ : live) reduction(| : edges)
    for (int64_t k = 0; k < t * t; k++) {
      int64_t r = k >> log, c = k & m;
      int count = 0;
      for (int64_t dr = -1; dr <= 1; dr++)
        for (int64_t dc = -1; dc <= 1; dc++) {
          if (dr == 0 && dc == 0)
            continue;
          int64_t rr = r + dr, cc = c + dc;
          int64_t tile = n[((rr >> log) + 1) * 3 + (cc >> log) + 1];
          count += cur[(tile << (2 * log)) + ((rr & m) << log) + (cc & m)];
        }
      uint8_t v = count == 3 ? 1 : count == 2 ? cur[base + k] : 0;
      next[base + k] = v;
      if (v) {
        live++;
        edges |= edge_bits(r, c, m);
      }
    }
    slot_live[j] = live;
    slot_edges[j] = (uint16_t)edges;
  }
}

// The cells of the active tiles from slot first on, which are new, all
// dead.
KERNEL void zero(int64_t log, int64_t first, int64_t nactive, int64_t cells,
    const int32_t *actives, uint8_t *next) {
  int64_t t = (int64_t)1 << log;
#pragma omp target teams distribute map(to : actives[0 : nactive]) \
    map(tofrom : next[0 : cells])
  for (int64_t j = first; j < nactive; j++) {
    int64_t base = (int64_t)actives[j] << (2 * log);
#pragma omp parallel for
    for (int64_t k = 0; k < t * t; k++)
      next[base + k] = 0;
  }
}

static int fail(const char *message) {
  printf("%s\n", message);
  return 1;
}

int main(void) {
  int64_t size = setting("LIFE_WORLD", 8192);
  int64_t tile = setting("LIFE_TILE", 64);
  int64_t soup = setting("LIFE_SOUP", 256);
  long steps = setting("LIFE_STEPS", 1000);
  int64_t log = 0;
  while (log < 62 && ((int64_t)1 << log) < tile)
    log++;
  int64_t side = tile > 0 ? size / tile : 0;
  int64_t capacity =
      setting("LIFE_CAPACITY", (side * side < 8192 ? side * side : 8192) + 1);
  long trace = setting("LIFE_TRACE", 0);
  if (tile < 4 || ((int64_t)1 << log) != tile || size < tile ||
      size % tile != 0 || soup < 0 || soup > size || steps < 2 || capacity < 2)
    return fail("LIFE_TILE must be a power of two of at least 4 that divides "
                "LIFE_WORLD, LIFE_SOUP from 0 to LIFE_WORLD, LIFE_STEPS at "
                "least 2, and LIFE_CAPACITY at least 2.");
  printf("Game of Life, C, sparse tiles\n");
  printf("World: %ld x %ld cells, tiles of %ld x %ld, %ld in the pool\n",
      (long)size, (long)size, (long)tile, (long)tile, (long)capacity - 1);
  printf("Soup: %ld x %ld cells at the centre\n", (long)soup, (long)soup);
  printf("Steps: %ld\n", steps);

  struct world w = {.tile = tile,
      .log = log,
      .side = side,
      .size = size,
      .capacity = capacity};
  int64_t cells = capacity * tile * tile;
  w.dir = calloc(side * side, sizeof(int32_t));
  w.pos = calloc(capacity, sizeof(int32_t));
  w.nb = calloc(capacity * 9, sizeof(int32_t));
  w.live = calloc(capacity, sizeof(int32_t));
  w.edges = calloc(capacity, sizeof(uint16_t));
  w.quiet = calloc(capacity, sizeof(uint8_t));
  w.actives = calloc(capacity, sizeof(int32_t));
  w.free = calloc(capacity, sizeof(int32_t));
  w.fate = calloc(capacity, sizeof(uint16_t));
  w.slot_nb = calloc(capacity * 9, sizeof(int32_t));
  w.slot_live = calloc(capacity, sizeof(int32_t));
  w.slot_edges = calloc(capacity, sizeof(uint16_t));
  uint8_t *a = calloc(cells, 1), *b = calloc(cells, 1);
  double *times = calloc(2 * steps, sizeof(double));
  long *counts = calloc(steps, sizeof(long));
  if (!w.dir || !w.pos || !w.nb || !w.live || !w.edges || !w.quiet ||
      !w.actives || !w.free || !w.fate || !w.slot_nb || !w.slot_live ||
      !w.slot_edges || !a || !b || !times || !counts)
    return fail("out of memory");
  // Tile 0 is the dead tile; the free list gives 1, 2 and so on first.
  for (int64_t j = 0; j < capacity - 1; j++)
    w.free[j] = (int32_t)(capacity - 1 - j);
  w.nfree = capacity - 1;

  // The soup's tiles, its cells, and each tile's edges all live, so that
  // the tiles around them start too.
  int64_t from = (size - soup) / 2;
  if (soup > 0) {
    int64_t first = from >> log, last = (from + soup - 1) >> log;
    for (int64_t y = first; y <= last; y++)
      for (int64_t x = first; x <= last; x++)
        start(&w, y * side + x);
  }
  init(&w, from, soup, a);
  for (int64_t j = 0; j < w.nactive; j++) {
    w.edges[w.actives[j]] = ALL_EDGES;
    w.fate[j] = ALL_EDGES;
  }
  grow(&w, w.nactive);
  slots(&w);
  long peak = w.nactive;

  int32_t *actives = w.actives, *slot_nb = w.slot_nb, *slot_live = w.slot_live;
  uint16_t *slot_edges = w.slot_edges;
  uint8_t *final = a;
  long done = 0;
#pragma omp target data map(tofrom : a[0 : cells], b[0 : cells]) \
    map(to : actives[0 : capacity], slot_nb[0 : capacity * 9]) \
    map(alloc : slot_live[0 : capacity], slot_edges[0 : capacity])
  {
    uint8_t *cur = a, *next = b;
    for (long s = 0; s < steps && !w.overflow; s++) {
      double t0 = now();
      int64_t n = w.nactive;
      step(log, n, cells, actives, slot_nb, cur, next, slot_live, slot_edges);
      double t1 = now();
#pragma omp target update from(slot_live[0 : n], slot_edges[0 : n])
      for (int64_t j = 0; j < n; j++) {
        w.live[actives[j]] = slot_live[j];
        w.edges[actives[j]] = slot_edges[j];
      }
      if (decide(&w)) {
        finish(&w);
        int64_t kept = w.nactive;
        grow(&w, kept);
        slots(&w);
        int64_t m = w.nactive;
#pragma omp target update to(actives[0 : m], slot_nb[0 : m * 9])
        if (m > kept)
          zero(log, kept, m, cells, actives, next);
      }
      double t2 = now();
      times[s] = t2 - t0;
      times[steps + s] = t2 - t1;
      counts[s] = w.nactive;
      peak = w.nactive > peak ? w.nactive : peak;
      uint8_t *swap = cur;
      cur = next;
      next = swap;
      done = s + 1;
    }
    final = cur;
  }
  if (w.overflow)
    return fail("The pool ran out of tiles: raise LIFE_CAPACITY.");

  // The live cells and the sum of their indices in the world.
  long population = 0, checksum = 0;
  for (int64_t j = 0; j < w.nactive; j++) {
    int64_t i = w.actives[j], p = w.pos[i];
    int64_t top = (p / side) << log, left = (p % side) << log;
    for (int64_t k = 0; k < tile * tile; k++) {
      long v = final[(i << (2 * log)) + k];
      population += v;
      checksum += v * ((top + (k >> log)) * size + left + (k & (tile - 1)));
    }
  }
  if (trace)
    for (long s = 0; s < done; s++)
      printf("step %ld: %ld tiles, %.6f s, bookkeeping %.6f s\n", s, counts[s],
          times[s], times[steps + s]);
  // Each step but the first, which includes starting the kernels.
  double min = times[1], max = times[1], sum = 0.0, keep = 0.0;
  for (long s = 1; s < steps; s++) {
    min = times[s] < min ? times[s] : min;
    max = times[s] > max ? times[s] : max;
    sum += times[s];
    keep += times[steps + s];
  }
  printf("Population: %ld\n", population);
  printf("Checksum: %ld\n", checksum);
  printf("Active tiles: %ld at most, %ld at the end\n", peak, (long)w.nactive);
  printf(
      "%-12s%-12s%-12s%-14s\n", "Min (sec)", "Max", "Average", "Bookkeeping");
  printf("%-12.6f%-12.6f%-12.6f%-14.6f\n", min, max, sum / (double)(steps - 1),
      keep / (double)(steps - 1));
  free(w.dir);
  free(w.pos);
  free(w.nb);
  free(w.live);
  free(w.edges);
  free(w.quiet);
  free(w.actives);
  free(w.free);
  free(w.fate);
  free(w.slot_nb);
  free(w.slot_live);
  free(w.slot_edges);
  free(a);
  free(b);
  free(times);
  free(counts);
  return 0;
}
