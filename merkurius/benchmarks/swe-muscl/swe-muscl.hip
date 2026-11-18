// The shallow water equations with wetting and drying on sparse tiles, in
// HIP, by second-order MUSCL-Hancock with HLLC fluxes: the bar that a GPU
// programmer would set for swe-muscl.m. The same world, tiles, scheme and
// bookkeeping as swe-muscl.c, whose functions it shares
// (swe.h); its kernels are HIP's, a block of 256 threads for each active
// tile, and its reductions the block's, in shared memory. See README.md.
//
// Built with -ffp-contract=off, as HIP's default fuses multiplies and adds,
// so that it gives swe-muscl.c's and swe-muscl.m's bits.
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SWE_DEVICE __host__ __device__
#define SWE_CONST static constexpr
#include "swe.h"

enum { THREADS = 256 };

static long setting(const char *name, long fallback) {
  const char *s = getenv(name);
  return s && *s ? strtol(s, NULL, 10) : fallback;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static void check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    printf("%s: %s\n", what, hipGetErrorString(e));
    exit(1);
  }
}

// The block's reductions of its threads' wet cells, wet edges and largest
// speed, into thread 0's.
__device__ static void reduce(int32_t *wet, int *edges, double *fastest) {
  __shared__ int32_t wets[THREADS];
  __shared__ int edgess[THREADS];
  __shared__ double speeds[THREADS];
  int t = threadIdx.x;
  wets[t] = *wet;
  edgess[t] = *edges;
  speeds[t] = *fastest;
  __syncthreads();
  for (int half = THREADS / 2; half > 0; half /= 2) {
    if (t < half) {
      wets[t] += wets[t + half];
      edgess[t] |= edgess[t + half];
      speeds[t] = maxf(speeds[t], speeds[t + half]);
    }
    __syncthreads();
  }
  *wet = wets[0];
  *edges = edgess[0];
  *fastest = speeds[0];
}

// The active tiles' cells at the start, and each one's largest speed, by
// its slot: a block for each tile.
__global__ static void init(int64_t log, int64_t side, int64_t size, int kase,
    const int32_t *actives, const int32_t *pos, double *cur,
    double *slot_speed) {
  int64_t t = (int64_t)1 << log, tt = t * t;
  int64_t j = blockIdx.x, i = actives[j], p = pos[i];
  int64_t top = (p / side) << log, left = (p % side) << log;
  int32_t wet = 0;
  int edges = 0;
  double fastest = 0.0;
  for (int64_t k = threadIdx.x; k < tt; k += THREADS) {
    double h = start_depth(top + (k >> log), left + (k & (t - 1)), size, kase);
    cur[i * 3 * tt + k] = h;
    cur[i * 3 * tt + tt + k] = 0.0;
    cur[i * 3 * tt + 2 * tt + k] = 0.0;
    fastest = maxf(fastest, speed(h, 0.0, 0.0));
  }
  reduce(&wet, &edges, &fastest);
  if (threadIdx.x == 0)
    slot_speed[j] = fastest;
}

// The state of the cell at row r and column c of a tile, each from -1 to
// T, through the tile's neighbours n.
__device__ static inline void cell_at(const double *cur, const int32_t *n,
    int64_t log, int64_t r, int64_t c, double *h, double *hu, double *hv) {
  int64_t t = (int64_t)1 << log, m = t - 1, tt = t * t;
  int64_t tile = n[((r >> log) + 1) * 3 + (c >> log) + 1];
  int64_t at = tile * 3 * tt + ((r & m) << log) + (c & m);
  *h = cur[at];
  *hu = cur[at + tt];
  *hv = cur[at + 2 * tt];
}

// The state of that cell, at row top + r and column left + c of the world
// of side size, with its bed.
__device__ static inline struct state state_at(const double *cur,
    const int32_t *n, int64_t log, int64_t size, int64_t top, int64_t left,
    int64_t r, int64_t c) {
  struct state s;
  cell_at(cur, n, log, r, c, &s.h, &s.hu, &s.hv);
  s.b = bed(top + r, left + c, size);
  return s;
}

// The states at that cell's faces half a step of dt on, each from -1 to T,
// from it and its four neighbours.
__device__ static inline struct faces faces_at(const double *cur,
    const int32_t *n, int64_t log, int64_t size, int64_t top, int64_t left,
    int64_t r, int64_t c, double dt) {
  return predict(dt, state_at(cur, n, log, size, top, left, r, c),
      state_at(cur, n, log, size, top, left, r, c - 1),
      state_at(cur, n, log, size, top, left, r, c + 1),
      state_at(cur, n, log, size, top, left, r - 1, c),
      state_at(cur, n, log, size, top, left, r + 1, c));
}

// The cell at row r and column c of tile i after a step, nh, nhu and nhv,
// dried if it is too shallow, into next; and its part of the tile's wet
// cells, wet edges and largest speed.
__device__ static inline void finish_cell(int64_t log, int64_t i, int64_t r,
    int64_t c, double nh, double nhu, double nhv, double *next, int32_t *wet,
    int *edges, double *fastest) {
  int64_t t = (int64_t)1 << log, m = t - 1, tt = t * t;
  int64_t k = (r << log) + c;
  nh = maxf(nh, 0.0);
  if (nh <= DRY) {
    nhu = 0.0;
    nhv = 0.0;
  }
  next[i * 3 * tt + k] = nh;
  next[i * 3 * tt + tt + k] = nhu;
  next[i * 3 * tt + 2 * tt + k] = nhv;
  if (nh > DRY) {
    ++*wet;
    *edges |= edge_bits(r, c, m);
  }
  *fastest = maxf(*fastest, speed(nh, nhu, nhv));
}

// One step of dt: next's cells of the active tiles from cur's, and each
// one's wet cells, wet edges and largest speed, by its slot; a block for
// each tile. Each cell computes its faces' states and its neighbours'
// facing ones, and the fluxes at its four faces.
__global__ static void step(int64_t log, int64_t side, int64_t size, double dt,
    const int32_t *actives, const int32_t *pos, const int32_t *slot_nb,
    const double *cur, double *next, int32_t *slot_wet, uint16_t *slot_edges,
    double *slot_speed) {
  int64_t t = (int64_t)1 << log, m = t - 1, tt = t * t;
  int64_t j = blockIdx.x, i = actives[j], p = pos[i];
  int64_t top = (p / side) << log, left = (p % side) << log;
  const int32_t *n = slot_nb + j * 9;
  int32_t wet = 0;
  int edges = 0;
  double fastest = 0.0;
  for (int64_t k = threadIdx.x; k < tt; k += THREADS) {
    int64_t r = k >> log, c = k & m;
    double h, hu, hv;
    cell_at(cur, n, log, r, c, &h, &hu, &hv);
    struct faces p = faces_at(cur, n, log, size, top, left, r, c, dt);
    struct state w = faces_at(cur, n, log, size, top, left, r, c - 1, dt).e;
    struct state e = faces_at(cur, n, log, size, top, left, r, c + 1, dt).w;
    struct state o = faces_at(cur, n, log, size, top, left, r - 1, c, dt).s;
    struct state s = faces_at(cur, n, log, size, top, left, r + 1, c, dt).n;
    // Across x the normal momentum is hu; across y, hv.
    struct flux fw = hllc(w.h, w.hu, w.hv, w.b, p.w.h, p.w.hu, p.w.hv, p.w.b);
    struct flux fe = hllc(p.e.h, p.e.hu, p.e.hv, p.e.b, e.h, e.hu, e.hv, e.b);
    struct flux fn = hllc(o.h, o.hv, o.hu, o.b, p.n.h, p.n.hv, p.n.hu, p.n.b);
    struct flux fs = hllc(p.s.h, p.s.hv, p.s.hu, p.s.b, s.h, s.hv, s.hu, s.b);
    double nh, nhu, nhv;
    advance(dt, h, hu, hv, p, fe.m - fw.m, (fe.n + fe.pl) - (fw.n + fw.pr),
        fe.a - fw.a, fs.m - fn.m, (fs.n + fs.pl) - (fn.n + fn.pr), fs.a - fn.a,
        &nh, &nhu, &nhv);
    finish_cell(log, i, r, c, nh, nhu, nhv, next, &wet, &edges, &fastest);
  }
  reduce(&wet, &edges, &fastest);
  if (threadIdx.x == 0) {
    slot_wet[j] = wet;
    slot_edges[j] = (uint16_t)edges;
    slot_speed[j] = fastest;
  }
}

// The most cells of a tile a thread of step_faces updates, and the bytes of
// shared memory it takes for tiles of side t: four doubles for each face
// across x, which the faces across y then reuse.
enum { FACE_CELLS = 4 };
static int64_t faces_bytes(int64_t t) { return 4 * t * (t + 1) * 8; }

// The flux through each face between cells l and r of a tile, by HLLC of
// their states there half a step on, into four words of shared memory at
// f, each `count` apart: its mass, its along and its normal flux with the
// left's and with the right's pressure correction, which are what the
// cells either side take, in the order they take them.
__device__ static inline void store_flux(
    double *faces, int count, int f, struct flux x) {
  faces[f] = x.m;
  faces[count + f] = x.a;
  faces[2 * count + f] = x.n + x.pl;
  faces[3 * count + f] = x.n + x.pr;
}

// step, with each face's flux computed once rather than by both its cells,
// through shared memory: first the faces across x, whose differences each
// thread keeps for its cells, then those across y. The same operations in
// the same order, so the same bits. Each thread has at most FACE_CELLS
// cells, so tiles have at most 32 x 32 of them.
__global__ static void step_faces(int64_t log, int64_t side, int64_t size,
    double dt, const int32_t *actives, const int32_t *pos,
    const int32_t *slot_nb, const double *cur, double *next, int32_t *slot_wet,
    uint16_t *slot_edges, double *slot_speed) {
  extern __shared__ double faces[];
  int64_t t = (int64_t)1 << log, m = t - 1, tt = t * t;
  int64_t j = blockIdx.x, i = actives[j], p = pos[i];
  int64_t top = (p / side) << log, left = (p % side) << log;
  const int32_t *n = slot_nb + j * 9;
  int count = (int)(t * (t + 1));
  // Across x: face f of row r is between columns f - 1 and f.
  for (int f = threadIdx.x; f < count; f += THREADS) {
    int64_t r = f / (t + 1), c = f % (t + 1);
    struct state l = faces_at(cur, n, log, size, top, left, r, c - 1, dt).e;
    struct state o = faces_at(cur, n, log, size, top, left, r, c, dt).w;
    store_flux(faces, count, f,
        hllc(l.h, l.hu, l.hv, l.b, o.h, o.hu, o.hv, o.b));
  }
  __syncthreads();
  double dm[FACE_CELLS], dn[FACE_CELLS], da[FACE_CELLS];
  for (int q = 0; q < FACE_CELLS; q++) {
    int64_t k = threadIdx.x + q * THREADS;
    if (k < tt) {
      int w = (int)((k >> log) * (t + 1) + (k & m));
      dm[q] = faces[w + 1] - faces[w];
      dn[q] = faces[2 * count + w + 1] - faces[3 * count + w];
      da[q] = faces[count + w + 1] - faces[count + w];
    }
  }
  __syncthreads();
  // Across y: face f of column c is between rows f / t - 1 and f / t.
  for (int f = threadIdx.x; f < count; f += THREADS) {
    int64_t r = f >> log, c = f & m;
    struct state l = faces_at(cur, n, log, size, top, left, r - 1, c, dt).s;
    struct state o = faces_at(cur, n, log, size, top, left, r, c, dt).n;
    store_flux(faces, count, f,
        hllc(l.h, l.hv, l.hu, l.b, o.h, o.hv, o.hu, o.b));
  }
  __syncthreads();
  int32_t wet = 0;
  int edges = 0;
  double fastest = 0.0;
  for (int q = 0; q < FACE_CELLS; q++) {
    int64_t k = threadIdx.x + q * THREADS;
    if (k < tt) {
      int64_t r = k >> log, c = k & m;
      int up = (int)k, down = (int)(k + t);
      double h, hu, hv;
      cell_at(cur, n, log, r, c, &h, &hu, &hv);
      struct faces pf = faces_at(cur, n, log, size, top, left, r, c, dt);
      double nh, nhu, nhv;
      advance(dt, h, hu, hv, pf, dm[q], dn[q], da[q],
          faces[down] - faces[up], faces[2 * count + down] - faces[3 * count + up],
          faces[count + down] - faces[count + up], &nh, &nhu, &nhv);
      finish_cell(log, i, r, c, nh, nhu, nhv, next, &wet, &edges, &fastest);
    }
  }
  reduce(&wet, &edges, &fastest);
  if (threadIdx.x == 0) {
    slot_wet[j] = wet;
    slot_edges[j] = (uint16_t)edges;
    slot_speed[j] = fastest;
  }
}

// The cells of the active tiles from slot first on, which are new, all dry:
// a block for each.
__global__ static void zero(
    int64_t log, int64_t first, const int32_t *actives, double *next) {
  int64_t tt = (int64_t)1 << (2 * log);
  int64_t base = (int64_t)actives[first + blockIdx.x] * 3 * tt;
  for (int64_t k = threadIdx.x; k < 3 * tt; k += THREADS)
    next[base + k] = 0.0;
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
  // The step's kernel: 0 step; 2 step_faces where its tiles fit in the
  // shared memory a block may have, otherwise step.
  long choice = setting("SWE_KERNEL", 0);
  int faced = choice == 2 && tile * tile <= FACE_CELLS * THREADS &&
      faces_bytes(tile) <= 65536;
  if (tile < 4 || ((int64_t)1 << log) != tile || size < tile ||
      size % tile != 0 || steps < 2 || capacity < 2 || kase < 0 || kase > 1)
    return fail("SWE_TILE must be a power of two of at least 4 that divides "
                "SWE_WORLD, SWE_STEPS at least 2, SWE_CAPACITY at least 2, "
                "and SWE_CASE 0 or 1.");
  printf("Shallow water, HIP, sparse tiles\n");
  printf("World: %ld x %ld cells, tiles of %ld x %ld, %ld in the pool\n",
      (long)size, (long)size, (long)tile, (long)tile, (long)capacity - 1);
  printf("Case: %s\n", kase == 0 ? "a column in a bowl" : "Thacker's bowl");
  printf("Steps: %ld\n", steps);

  struct world w = {};
  w.tile = tile;
  w.log = log;
  w.side = side;
  w.size = size;
  w.capacity = capacity;
  int64_t tt = tile * tile, cells = capacity * 3 * tt;
  w.dir = (int32_t *)calloc(side * side, sizeof(int32_t));
  w.pos = (int32_t *)calloc(capacity, sizeof(int32_t));
  w.nb = (int32_t *)calloc(capacity * 9, sizeof(int32_t));
  w.wet = (int32_t *)calloc(capacity, sizeof(int32_t));
  w.edges = (uint16_t *)calloc(capacity, sizeof(uint16_t));
  w.quiet = (uint8_t *)calloc(capacity, sizeof(uint8_t));
  w.actives = (int32_t *)calloc(capacity, sizeof(int32_t));
  w.free = (int32_t *)calloc(capacity, sizeof(int32_t));
  w.fate = (uint16_t *)calloc(capacity, sizeof(uint16_t));
  w.slot_nb = (int32_t *)calloc(capacity * 9, sizeof(int32_t));
  w.slot_wet = (int32_t *)calloc(capacity, sizeof(int32_t));
  w.slot_edges = (uint16_t *)calloc(capacity, sizeof(uint16_t));
  w.slot_speed = (double *)calloc(capacity, sizeof(double));
  double *final = (double *)malloc(cells * sizeof(double));
  double *times = (double *)calloc(3 * steps, sizeof(double));
  long *counts = (long *)calloc(steps, sizeof(long));
  if (!w.dir || !w.pos || !w.nb || !w.wet || !w.edges || !w.quiet ||
      !w.actives || !w.free || !w.fate || !w.slot_nb || !w.slot_wet ||
      !w.slot_edges || !w.slot_speed || !final || !times || !counts)
    return fail("out of memory");
  // The GPU's: the two pools, and what it reads and writes by slot.
  double *a, *b, *d_speed;
  int32_t *d_actives, *d_pos, *d_nb, *d_wet;
  uint16_t *d_edges;
  check(hipMalloc(&a, cells * sizeof(double)), "hipMalloc");
  check(hipMalloc(&b, cells * sizeof(double)), "hipMalloc");
  check(hipMalloc(&d_actives, capacity * sizeof(int32_t)), "hipMalloc");
  check(hipMalloc(&d_pos, capacity * sizeof(int32_t)), "hipMalloc");
  check(hipMalloc(&d_nb, capacity * 9 * sizeof(int32_t)), "hipMalloc");
  check(hipMalloc(&d_wet, capacity * sizeof(int32_t)), "hipMalloc");
  check(hipMalloc(&d_edges, capacity * sizeof(uint16_t)), "hipMalloc");
  check(hipMalloc(&d_speed, capacity * sizeof(double)), "hipMalloc");
  check(hipMemset(a, 0, cells * sizeof(double)), "hipMemset");
  check(hipMemset(b, 0, cells * sizeof(double)), "hipMemset");
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
  check(hipMemcpy(d_actives, w.actives, first * sizeof(int32_t),
            hipMemcpyHostToDevice),
      "hipMemcpy");
  check(hipMemcpy(
            d_pos, w.pos, capacity * sizeof(int32_t), hipMemcpyHostToDevice),
      "hipMemcpy");
  init<<<first, THREADS>>>(log, side, size, kase, d_actives, d_pos, a, d_speed);
  check(hipMemcpy(w.slot_speed, d_speed, first * sizeof(double),
            hipMemcpyDeviceToHost),
      "init");
  double fastest = 0.0;
  for (int64_t j = 0; j < first; j++) {
    fastest = maxf(fastest, w.slot_speed[j]);
    w.edges[w.actives[j]] = ALL_EDGES;
    w.fate[j] = ALL_EDGES;
  }
  grow(&w, first);
  slots(&w);
  check(hipMemcpy(d_actives, w.actives, w.nactive * sizeof(int32_t),
            hipMemcpyHostToDevice),
      "hipMemcpy");
  check(hipMemcpy(
            d_pos, w.pos, capacity * sizeof(int32_t), hipMemcpyHostToDevice),
      "hipMemcpy");
  check(hipMemcpy(d_nb, w.slot_nb, w.nactive * 9 * sizeof(int32_t),
            hipMemcpyHostToDevice),
      "hipMemcpy");
  long peak = w.nactive;

  double *cur = a, *next = b, time = 0.0;
  long done = 0;
  hipEvent_t began, ended;
  check(hipEventCreate(&began), "hipEventCreate");
  check(hipEventCreate(&ended), "hipEventCreate");
  for (long s = 0; s < steps && !w.overflow; s++) {
    double t0 = now();
    int64_t n = w.nactive;
    double dt = COURANT / maxf(fastest, DRY);
    check(hipEventRecord(began), "hipEventRecord");
    if (faced)
      step_faces<<<n, THREADS, faces_bytes(tile)>>>(log, side, size, dt,
          d_actives, d_pos, d_nb, cur, next, d_wet, d_edges, d_speed);
    else
      step<<<n, THREADS>>>(log, side, size, dt, d_actives, d_pos, d_nb, cur,
          next, d_wet, d_edges, d_speed);
    check(hipEventRecord(ended), "hipEventRecord");
    check(hipDeviceSynchronize(), "step");
    double t1 = now();
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, began, ended), "hipEventElapsedTime");
    times[2 * steps + s] = 1.0e-3 * (double)ms;
    time = time + dt;
    check(hipMemcpy(
              w.slot_wet, d_wet, n * sizeof(int32_t), hipMemcpyDeviceToHost),
        "hipMemcpy");
    check(hipMemcpy(w.slot_edges, d_edges, n * sizeof(uint16_t),
              hipMemcpyDeviceToHost),
        "hipMemcpy");
    check(hipMemcpy(
              w.slot_speed, d_speed, n * sizeof(double), hipMemcpyDeviceToHost),
        "hipMemcpy");
    fastest = 0.0;
    for (int64_t j = 0; j < n; j++) {
      w.wet[w.actives[j]] = w.slot_wet[j];
      w.edges[w.actives[j]] = w.slot_edges[j];
      fastest = maxf(fastest, w.slot_speed[j]);
    }
    if (decide(&w)) {
      finish(&w);
      int64_t kept = w.nactive;
      grow(&w, kept);
      slots(&w);
      int64_t m = w.nactive;
      check(hipMemcpy(d_actives, w.actives, m * sizeof(int32_t),
                hipMemcpyHostToDevice),
          "hipMemcpy");
      check(hipMemcpy(d_pos, w.pos, capacity * sizeof(int32_t),
                hipMemcpyHostToDevice),
          "hipMemcpy");
      check(hipMemcpy(d_nb, w.slot_nb, m * 9 * sizeof(int32_t),
                hipMemcpyHostToDevice),
          "hipMemcpy");
      if (m > kept) {
        zero<<<m - kept, THREADS>>>(log, kept, d_actives, next);
        check(hipDeviceSynchronize(), "zero");
      }
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
  if (w.overflow)
    return fail("The pool ran out of tiles: raise SWE_CAPACITY.");
  check(hipMemcpy(final, cur, cells * sizeof(double), hipMemcpyDeviceToHost),
      "hipMemcpy");

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
  // Each step but the first, which includes loading the kernels.
  double min = times[1], max = times[1], sum = 0.0, keep = 0.0, kernel = 0.0;
  for (long s = 1; s < steps; s++) {
    min = times[s] < min ? times[s] : min;
    max = times[s] > max ? times[s] : max;
    sum += times[s];
    keep += times[steps + s];
    kernel += times[2 * steps + s];
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
  // The step's kernel alone, by HIP's events, on average.
  printf("Kernel (%s): %.6f s\n", faced ? "faces" : "step",
      kernel / (double)(steps - 1));
  check(hipEventDestroy(began), "hipEventDestroy");
  check(hipEventDestroy(ended), "hipEventDestroy");
  check(hipFree(a), "hipFree");
  check(hipFree(b), "hipFree");
  check(hipFree(d_actives), "hipFree");
  check(hipFree(d_pos), "hipFree");
  check(hipFree(d_nb), "hipFree");
  check(hipFree(d_wet), "hipFree");
  check(hipFree(d_edges), "hipFree");
  check(hipFree(d_speed), "hipFree");
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
  free(final);
  free(times);
  free(counts);
  return 0;
}
