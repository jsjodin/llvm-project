// A deep Mandelbrot zoom in C with OpenMP, the bar for mandel.m and the
// places of mandel-places.m (docs/INPROGRESS.md, step 4 of the demo): a
// kernel a frame over the frame's pixels, whose rows the GPU's hardware
// balances across its compute units. See README.md.
//
// Frame f is MANDEL_SIZE pixels square (2048), of width 3 * 0.5^f, centred
// on a point of the set's boundary, with an escape limit of 256 + 64 * f,
// for MANDEL_FRAMES frames (40); MANDEL_CENTRE=x,y moves the centre, as for
// the work-stealing example's (examples/work-stealing). Each pixel's iterations
// follow one order of operations, without fused multiplies and adds
// (-ffp-contract=off, as merkc does not fuse), so that every program and
// mandel.py give the same counts. MANDEL_TRACE=1 prints each frame's line.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#pragma STDC FP_CONTRACT OFF

// The kernel is a separate function, as in mandel.m.
#define KERNEL __attribute__((noinline)) static

// The centre unless MANDEL_CENTRE gives another.
static const double kX0 = -0.743643887037151;
static const double kY0 = 0.131825904205330;

static long setting(const char *name, long fallback) {
  const char *s = getenv(name);
  return s && *s ? strtol(s, NULL, 10) : fallback;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

// The iterations before z escapes from c, at most limit.
#pragma omp declare target
static inline int64_t escape(double cx, double cy, int64_t limit) {
  double x = 0.0, y = 0.0;
  for (int64_t n = 0; n < limit; n++) {
    double x2 = x * x, y2 = y * y;
    if (x2 + y2 > 4.0)
      return n;
    y = (2.0 * x) * y + cy;
    x = x2 - y2 + cx;
  }
  return limit;
}
#pragma omp end declare target

// A frame's iterations: its pixels' sum.
KERNEL int64_t frame(
    double x0, double y0, int64_t size, double step, int64_t limit) {
  int64_t total = 0;
  int64_t half = size / 2;
#pragma omp target teams distribute parallel for reduction(+ : total) \
    map(tofrom : total)
  for (int64_t k = 0; k < size * size; k++) {
    int64_t py = k / size, px = k % size;
    double cx = x0 + (double)(px - half) * step;
    double cy = y0 + (double)(py - half) * step;
    total += escape(cx, cy, limit);
  }
  return total;
}

int main(void) {
  int64_t size = setting("MANDEL_SIZE", 2048);
  long frames = setting("MANDEL_FRAMES", 40);
  long trace = setting("MANDEL_TRACE", 0);
  double x0 = kX0, y0 = kY0;
  const char *centre = getenv("MANDEL_CENTRE");
  char *end = NULL;
  if (centre && *centre) {
    x0 = strtod(centre, &end);
    if (*end == ',')
      y0 = strtod(end + 1, &end);
  }
  if (size < 2 || frames < 1) {
    printf("MANDEL_SIZE must be at least 2 and MANDEL_FRAMES at least 1.\n");
    return 1;
  }
  if (centre && *centre && *end != '\0') {
    printf("MANDEL_CENTRE must be two numbers, x,y.\n");
    return 1;
  }
  printf("Mandelbrot zoom, C\n");
  printf("Frames: %ld of %ld x %ld pixels\n", frames, (long)size, (long)size);
  double width = 3.0, start = now();
  long total = 0, checksum = 0;
  for (long f = 0; f < frames; f++) {
    int64_t limit = 256 + 64 * f;
    double t0 = now();
    long iterations = frame(x0, y0, size, width / (double)size, limit);
    double t1 = now();
    if (trace)
      printf("frame %ld: limit %ld, %ld iterations, %.6f s\n", f, (long)limit,
          iterations, t1 - t0);
    total += iterations;
    checksum += (f + 1) * iterations;
    width = width * 0.5;
  }
  double time = now() - start;
  printf("Iterations: %ld\n", total);
  printf("Checksum: %ld\n", checksum);
  printf("%-12s%-14s\n", "Time (sec)", "Giterations/s");
  printf("%-12.6f%-14.3f\n", time, (double)total / time * 1.0e-9);
  return 0;
}
