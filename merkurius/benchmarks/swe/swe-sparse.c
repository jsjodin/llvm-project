// The shallow water equations with wetting and drying on sparse tiles, in C
// with OpenMP offload: the bar for swe-sparse.m (docs/INPROGRESS.md, A
// showcase with more work per tile), as a C programmer would write it for a
// GPU. See README.md.
//
// The host keeps the tiles' tables and does the bookkeeping, as
// life-sparse.c does; the GPU keeps the cells. Each step is one kernel, a
// team for each active tile and its threads over the tile's cells, which
// reduce the tile's wet cells, wet edges and largest wave speed. The host
// copies those back, takes the next time step from the largest speed,
// decides which tiles end and start, and only if some do copies the active
// tiles and their neighbours to the GPU and zeroes the new tiles' cells
// there, a second kernel.
//
// - The world is SWE_WORLD cells square (2048), of side 1 m, in tiles of
//   SWE_TILE cells square (32), a power of two that divides it. Beyond its
//   edge, and in a tile that is not active, cells are dry.
// - A tile's cells are 3 T^2 doubles: its depths h by rows, then its
//   momenta hu, then hv. Two pools of SWE_CAPACITY tiles, which swap after
//   each step; tile 0 is the dry tile, all zero.
// - The scheme: an HLL flux at each face of a cell, with hydrostatic
//   reconstruction over the bed b = D0 (r^2 / R^2 - 1), computed where
//   needed; explicit Euler, the step 0.45 / max(|u| + |v| + 2 sqrt(g h)).
// - SWE_CASE 0: a column of water collapses and sloshes across the bowl;
//   1: Thacker's oscillating paraboloid, against its exact solution.
// - The checksum is the sum, modulo 2^64, over the active cells of
//   (bits(h) + 3 bits(hu) + 5 bits(hv)) (2 g + 1), g the cell's index in
//   the world and bits a double's, -0.0 as 0.0: the same in any order.
//
// Every operation is in the order swe-sparse.m has it, without fused
// multiplies and adds (-ffp-contract=off), so that the programs give the
// same bits.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// The kernels are separate functions, as in swe-sparse.m.
#define KERNEL __attribute__((noinline)) static

static long setting(const char *name, long fallback) {
  const char *s = getenv(name);
  return s && *s ? strtol(s, NULL, 10) : fallback;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

// The scheme's functions are the GPU's too.
#define SWE_DEVICE
#pragma omp declare target
#include "swe.h"

// The state of the cell at row r and column c of a tile, each from -1 to
// T, through the tile's neighbours n.
static inline void cell_at(const double *cur, const int32_t *n, int64_t log,
    int64_t r, int64_t c, double *h, double *hu, double *hv) {
  int64_t t = (int64_t)1 << log, m = t - 1, tt = t * t;
  int64_t tile = n[((r >> log) + 1) * 3 + (c >> log) + 1];
  int64_t at = tile * 3 * tt + ((r & m) << log) + (c & m);
  *h = cur[at];
  *hu = cur[at + tt];
  *hv = cur[at + 2 * tt];
}
#pragma omp end declare target

// The active tiles' cells at the start, and each one's largest speed, by
// its slot.
KERNEL void init(int64_t log, int64_t side, int64_t size, int kase,
    int64_t nactive, int64_t capacity, const int32_t *actives,
    const int32_t *pos, double *cur, double *slot_speed) {
  int64_t t = (int64_t)1 << log, tt = t * t, cells = capacity * 3 * tt;
#pragma omp target teams distribute map( \
        to : actives[0 : nactive], pos[0 : capacity]) \
    map(tofrom : cur[0 : cells], slot_speed[0 : nactive])
  for (int64_t j = 0; j < nactive; j++) {
    int64_t i = actives[j], p = pos[i];
    int64_t top = (p / side) << log, left = (p % side) << log;
    double fastest = 0.0;
#pragma omp parallel for reduction(max : fastest)
    for (int64_t k = 0; k < tt; k++) {
      double h =
          start_depth(top + (k >> log), left + (k & (t - 1)), size, kase);
      cur[i * 3 * tt + k] = h;
      cur[i * 3 * tt + tt + k] = 0.0;
      cur[i * 3 * tt + 2 * tt + k] = 0.0;
      fastest = maxf(fastest, speed(h, 0.0, 0.0));
    }
    slot_speed[j] = fastest;
  }
}

// One step of dt: next's cells of the active tiles from cur's, and each
// one's wet cells, wet edges and largest speed, by its slot.
KERNEL void step(int64_t log, int64_t side, int64_t size, double dt,
    int64_t nactive, int64_t cells, const int32_t *actives, const int32_t *pos,
    int64_t capacity, const int32_t *slot_nb, const double *cur, double *next,
    int32_t *slot_wet, uint16_t *slot_edges, double *slot_speed) {
  int64_t t = (int64_t)1 << log, m = t - 1, tt = t * t;
#pragma omp target teams distribute map(to : actives[0 : nactive], \
        pos[0 : capacity], slot_nb[0 : nactive * 9], cur[0 : cells]) \
    map(tofrom : next[0 : cells], slot_wet[0 : nactive], \
            slot_edges[0 : nactive], slot_speed[0 : nactive])
  for (int64_t j = 0; j < nactive; j++) {
    int64_t i = actives[j], p = pos[i];
    int64_t top = (p / side) << log, left = (p % side) << log;
    const int32_t *n = slot_nb + j * 9;
    int32_t wet = 0;
    int edges = 0;
    double fastest = 0.0;
#pragma omp parallel for reduction(+ : wet) reduction(| : edges) \
    reduction(max : fastest)
    for (int64_t k = 0; k < tt; k++) {
      int64_t r = k >> log, c = k & m;
      int64_t y = top + r, x = left + c;
      double h, hu, hv, hw, huw, hvw, he, hue, hve, hn, hun, hvn, hs, hus, hvs;
      cell_at(cur, n, log, r, c, &h, &hu, &hv);
      cell_at(cur, n, log, r, c - 1, &hw, &huw, &hvw);
      cell_at(cur, n, log, r, c + 1, &he, &hue, &hve);
      cell_at(cur, n, log, r - 1, c, &hn, &hun, &hvn);
      cell_at(cur, n, log, r + 1, c, &hs, &hus, &hvs);
      double b = bed(y, x, size);
      double bw = bed(y, x - 1, size), be = bed(y, x + 1, size);
      double bn = bed(y - 1, x, size), bs = bed(y + 1, x, size);
      // Across x the normal momentum is hu; across y, hv.
      struct flux fw = hll(hw, huw, hvw, bw, h, hu, hv, b);
      struct flux fe = hll(h, hu, hv, b, he, hue, hve, be);
      struct flux fn = hll(hn, hvn, hun, bn, h, hv, hu, b);
      struct flux fs = hll(h, hv, hu, b, hs, hvs, hus, bs);
      double nh = h - dt * ((fe.m - fw.m) + (fs.m - fn.m));
      double nhu =
          hu - dt * (((fe.n + fe.pl) - (fw.n + fw.pr)) + (fs.a - fn.a));
      double nhv =
          hv - dt * ((fe.a - fw.a) + ((fs.n + fs.pl) - (fn.n + fn.pr)));
      nh = maxf(nh, 0.0);
      if (nh <= DRY) {
        nhu = 0.0;
        nhv = 0.0;
      }
      next[i * 3 * tt + k] = nh;
      next[i * 3 * tt + tt + k] = nhu;
      next[i * 3 * tt + 2 * tt + k] = nhv;
      if (nh > DRY) {
        wet++;
        edges |= edge_bits(r, c, m);
      }
      fastest = maxf(fastest, speed(nh, nhu, nhv));
    }
    slot_wet[j] = wet;
    slot_edges[j] = (uint16_t)edges;
    slot_speed[j] = fastest;
  }
}

// The cells of the active tiles from slot first on, which are new, all
// dry.
KERNEL void zero(int64_t log, int64_t first, int64_t nactive, int64_t cells,
    const int32_t *actives, double *next) {
  int64_t tt = (int64_t)1 << (2 * log);
#pragma omp target teams distribute map(to : actives[0 : nactive]) \
    map(tofrom : next[0 : cells])
  for (int64_t j = first; j < nactive; j++) {
    int64_t base = (int64_t)actives[j] * 3 * tt;
#pragma omp parallel for
    for (int64_t k = 0; k < 3 * tt; k++)
      next[base + k] = 0.0;
  }
}

static int fail(const char *message) {
  printf("%s\n", message);
  return 1;
}

int main(void) {
  int64_t size = setting("SWE_WORLD", 2048);
  int64_t tile = setting("SWE_TILE", 32);
  long steps = setting("SWE_STEPS", 1000);
  int kase = (int)setting("SWE_CASE", 0);
  int64_t log = 0;
  while (log < 62 && ((int64_t)1 << log) < tile)
    log++;
  int64_t side = tile > 0 ? size / tile : 0;
  int64_t capacity =
      setting("SWE_CAPACITY", (side * side < 4096 ? side * side : 4096) + 1);
  long trace = setting("SWE_TRACE", 0);
  if (tile < 4 || ((int64_t)1 << log) != tile || size < tile ||
      size % tile != 0 || steps < 2 || capacity < 2 || kase < 0 || kase > 1)
    return fail("SWE_TILE must be a power of two of at least 4 that divides "
                "SWE_WORLD, SWE_STEPS at least 2, SWE_CAPACITY at least 2, "
                "and SWE_CASE 0 or 1.");
  printf("Shallow water, C, sparse tiles\n");
  printf("World: %ld x %ld cells, tiles of %ld x %ld, %ld in the pool\n",
      (long)size, (long)size, (long)tile, (long)tile, (long)capacity - 1);
  printf("Case: %s\n", kase == 0 ? "a column in a bowl" : "Thacker's bowl");
  printf("Steps: %ld\n", steps);

  struct world w = {.tile = tile,
      .log = log,
      .side = side,
      .size = size,
      .capacity = capacity};
  int64_t tt = tile * tile, cells = capacity * 3 * tt;
  w.dir = calloc(side * side, sizeof(int32_t));
  w.pos = calloc(capacity, sizeof(int32_t));
  w.nb = calloc(capacity * 9, sizeof(int32_t));
  w.wet = calloc(capacity, sizeof(int32_t));
  w.edges = calloc(capacity, sizeof(uint16_t));
  w.quiet = calloc(capacity, sizeof(uint8_t));
  w.actives = calloc(capacity, sizeof(int32_t));
  w.free = calloc(capacity, sizeof(int32_t));
  w.fate = calloc(capacity, sizeof(uint16_t));
  w.slot_nb = calloc(capacity * 9, sizeof(int32_t));
  w.slot_wet = calloc(capacity, sizeof(int32_t));
  w.slot_edges = calloc(capacity, sizeof(uint16_t));
  w.slot_speed = calloc(capacity, sizeof(double));
  double *a = calloc(cells, sizeof(double)), *b = calloc(cells, sizeof(double));
  double *times = calloc(2 * steps, sizeof(double));
  long *counts = calloc(steps, sizeof(long));
  if (!w.dir || !w.pos || !w.nb || !w.wet || !w.edges || !w.quiet ||
      !w.actives || !w.free || !w.fate || !w.slot_nb || !w.slot_wet ||
      !w.slot_edges || !w.slot_speed || !a || !b || !times || !counts)
    return fail("out of memory");
  // Tile 0 is the dry tile; the free list gives 1, 2 and so on first.
  for (int64_t j = 0; j < capacity - 1; j++)
    w.free[j] = (int32_t)(capacity - 1 - j);
  w.nfree = capacity - 1;

  // The tiles over the wet region's bounding square, and each tile's edges
  // all wet, so that the tiles around them start too.
  double half = kase == 0 ? (double)size / 16.0 : 0.45 * (double)size;
  double cx = 0.5 * (double)size + (kase == 0 ? 0.2 * (double)size : 0.0);
  double cy = 0.5 * (double)size;
  int64_t x0 = (int64_t)floor(cx - half), x1 = (int64_t)ceil(cx + half);
  int64_t y0 = (int64_t)floor(cy - half), y1 = (int64_t)ceil(cy + half);
  x0 = x0 < 0 ? 0 : x0;
  y0 = y0 < 0 ? 0 : y0;
  x1 = x1 > size - 1 ? size - 1 : x1;
  y1 = y1 > size - 1 ? size - 1 : y1;
  for (int64_t y = y0 >> log; y <= y1 >> log; y++)
    for (int64_t x = x0 >> log; x <= x1 >> log; x++)
      start(&w, y * side + x);
  if (w.overflow)
    return fail("The pool ran out of tiles: raise SWE_CAPACITY.");
  int64_t first = w.nactive;
  init(log, side, size, kase, first, capacity, w.actives, w.pos, a,
      w.slot_speed);
  double fastest = 0.0;
  for (int64_t j = 0; j < first; j++) {
    fastest = maxf(fastest, w.slot_speed[j]);
    w.edges[w.actives[j]] = ALL_EDGES;
    w.fate[j] = ALL_EDGES;
  }
  grow(&w, first);
  slots(&w);
  long peak = w.nactive;

  int32_t *actives = w.actives, *pos = w.pos, *slot_nb = w.slot_nb,
          *slot_wet = w.slot_wet;
  uint16_t *slot_edges = w.slot_edges;
  double *slot_speed = w.slot_speed;
  double *final = a, time = 0.0;
  long done = 0;
#pragma omp target data map(tofrom : a[0 : cells], b[0 : cells]) \
    map(to : actives[0 : capacity], pos[0 : capacity], \
            slot_nb[0 : capacity * 9]) map(alloc : slot_wet[0 : capacity], \
            slot_edges[0 : capacity], slot_speed[0 : capacity])
  {
    double *cur = a, *next = b;
    for (long s = 0; s < steps && !w.overflow; s++) {
      double t0 = now();
      int64_t n = w.nactive;
      double dt = COURANT / maxf(fastest, DRY);
      step(log, side, size, dt, n, cells, actives, pos, capacity, slot_nb, cur,
          next, slot_wet, slot_edges, slot_speed);
      double t1 = now();
      time = time + dt;
#pragma omp target update from( \
        slot_wet[0 : n], slot_edges[0 : n], slot_speed[0 : n])
      fastest = 0.0;
      for (int64_t j = 0; j < n; j++) {
        w.wet[actives[j]] = slot_wet[j];
        w.edges[actives[j]] = slot_edges[j];
        fastest = maxf(fastest, slot_speed[j]);
      }
      if (decide(&w)) {
        finish(&w);
        int64_t kept = w.nactive;
        grow(&w, kept);
        slots(&w);
        int64_t m = w.nactive;
#pragma omp target update to( \
        actives[0 : m], pos[0 : capacity], slot_nb[0 : m * 9])
        if (m > kept)
          zero(log, kept, m, cells, actives, next);
      }
      double t2 = now();
      times[s] = t2 - t0;
      times[steps + s] = t2 - t1;
      counts[s] = w.nactive;
      peak = w.nactive > peak ? w.nactive : peak;
      double *swap = cur;
      cur = next;
      next = swap;
      done = s + 1;
    }
    final = cur;
  }
  if (w.overflow)
    return fail("The pool ran out of tiles: raise SWE_CAPACITY.");

  // The wet cells, the volume, the checksum and, for Thacker's bowl, the
  // error against the exact depth.
  long wet = 0;
  uint64_t checksum = 0;
  double volume = 0.0, error = 0.0, exact = 0.0;
  for (int64_t j = 0; j < w.nactive; j++) {
    int64_t i = w.actives[j], p = w.pos[i];
    int64_t top = (p / side) << log, left = (p % side) << log;
    for (int64_t k = 0; k < tt; k++) {
      double h = final[i * 3 * tt + k];
      wet += h > DRY;
      volume += h;
      int64_t y = top + (k >> log), x = left + (k & (tile - 1));
      checksum += (bits(h) + 3 * bits(final[i * 3 * tt + tt + k]) +
                      5 * bits(final[i * 3 * tt + 2 * tt + k])) *
          (2 * (uint64_t)(y * size + x) + 1);
      if (kase == 1) {
        double e = thacker(y, x, size, time);
        error += fabs(h - e);
        exact += e;
      }
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
  printf("Time: %.9f s\n", time);
  printf("Wet cells: %ld\n", wet);
  printf("Volume: %.6e m^3\n", volume);
  printf("Checksum: %016llx\n", (unsigned long long)checksum);
  if (kase == 1)
    printf("Error: %.4e\n", error / exact);
  printf("Active tiles: %ld at most, %ld at the end\n", peak, (long)w.nactive);
  printf(
      "%-12s%-12s%-12s%-14s\n", "Min (sec)", "Max", "Average", "Bookkeeping");
  printf("%-12.6f%-12.6f%-12.6f%-14.6f\n", min, max, sum / (double)(steps - 1),
      keep / (double)(steps - 1));
  free(w.dir);
  free(w.pos);
  free(w.nb);
  free(w.wet);
  free(w.edges);
  free(w.quiet);
  free(w.actives);
  free(w.free);
  free(w.fate);
  free(w.slot_nb);
  free(w.slot_wet);
  free(w.slot_edges);
  free(w.slot_speed);
  free(a);
  free(b);
  free(times);
  free(counts);
  return 0;
}
