// Conway's Game of Life of life.m in C with OpenMP, the bar that Merkurius
// is measured against: the same grid, start, kernels, maps, timing, check
// and output, statement for statement where C allows it. See README.md.
//
// CELL is the type of a cell: int64_t by default, as wide as Mercury's int,
// to compare like with like, and uint8_t (-DCELL=uint8_t) as C would
// usually have it, to show what the wider cells cost.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef CELL
#define CELL int64_t
#endif
#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

typedef CELL cell;

// The kernels are separate functions, as in life.m, so that clang sees the
// arrays only through their parameters there too.
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

// Whether the cell of index k, counting the grid's cells by rows from 0,
// starts alive, if it is in the soup: about 30% do, by a multiplicative hash
// of k.
#pragma omp declare target
static inline cell alive(int64_t k) {
  int64_t h = (k * 2654435761) & 4294967295;
  return (h >> 16) % 100 < 30 ? 1 : 0;
}
#pragma omp end declare target

// The cells alive at the start, in a square soup of side `soup` at the
// grid's centre, its rows and columns from..from + soup - 1.
KERNEL void init(int64_t n, int64_t soup, cell *a) {
  int64_t w = n + 2;
  int64_t from = (n - soup) / 2;
#pragma omp target teams distribute parallel for map(tofrom : a[0 : w * w])
  for (int64_t k = 0; k < n * n; k++) {
    int64_t row = k / n, column = k % n;
    int64_t c = (row + 1) * w + column + 1;
    a[c] = row >= from && row < from + soup && column >= from &&
            column < from + soup
        ? alive(k)
        : 0;
  }
}

// One step: b's cells from a's.
KERNEL void step(int64_t n, const cell *a, cell *b) {
  int64_t w = n + 2;
#pragma omp target teams distribute parallel for map(to : a[0 : w * w])      \
    map(tofrom : b[0 : w * w])
  for (int64_t k = 0; k < n * n; k++) {
    int64_t c = (k / n + 1) * w + k % n + 1;
    int64_t count = a[c - w - 1] + a[c - w] + a[c - w + 1] + a[c - 1] +
                    a[c + 1] + a[c + w - 1] + a[c + w] + a[c + w + 1];
    cell v;
    if (count == 3)
      v = 1;
    else if (count == 2)
      v = a[c];
    else
      v = 0;
    b[c] = v;
  }
}

int main(void) {
  int64_t n = setting("LIFE_SIZE", 4096);
  long steps = setting("LIFE_STEPS", 100);
  int64_t soup = setting("LIFE_SOUP", n);
  if (n < 1 || steps < 2 || soup < 0 || soup > n) {
    printf("LIFE_SIZE must be at least 1, LIFE_STEPS at least 2, and "
           "LIFE_SOUP from 0 to LIFE_SIZE.\n");
    return 1;
  }
  printf("Game of Life, C (" STRINGIFY(CELL) ")\n");
  printf("Grid: %ld x %ld cells, %ld bytes a cell\n", (long)n, (long)n,
         (long)sizeof(cell));
  printf("Steps: %ld\n", steps);
  if (soup < n)
    printf("Soup: %ld x %ld cells at the centre\n", (long)soup, (long)soup);
  int64_t w = n + 2;
  cell *a = calloc(w * w, sizeof(cell));
  cell *b = calloc(w * w, sizeof(cell));
  double *t = calloc(steps, sizeof(double));
  if (!a || !b || !t) {
    printf("out of memory\n");
    return 1;
  }
  init(n, soup, a);
  // The arrays stay on the device for the steps; each step's own maps find
  // them there and copy nothing.
  cell *final = a;
#pragma omp target data map(tofrom : a[0 : w * w], b[0 : w * w])
  {
    cell *cur = a, *next = b;
    for (long s = 0; s < steps; s++) {
      double t0 = now();
      step(n, cur, next);
      double t1 = now();
      t[s] = t1 - t0;
      cell *swap = cur;
      cur = next;
      next = swap;
    }
    final = cur;
  }
  long population = 0, checksum = 0;
  for (int64_t k = 0; k < n * n; k++) {
    int64_t c = (k / n + 1) * w + k % n + 1;
    population += final[c];
    checksum += final[c] * k;
  }
  // Each step but the first, which includes starting the kernels.
  double min = t[1], max = t[1], sum = 0.0;
  for (long s = 1; s < steps; s++) {
    min = t[s] < min ? t[s] : min;
    max = t[s] > max ? t[s] : max;
    sum += t[s];
  }
  double average = sum / (double)(steps - 1);
  double rate = (double)n * (double)n / min * 1.0e-9;
  printf("Population: %ld\n", population);
  printf("Checksum: %ld\n", checksum);
  printf("%-14s%-12s%-12s%-12s\n", "Gcells/sec", "Min (sec)", "Max",
         "Average");
  printf("%-14.3f%-12.6f%-12.6f%-12.6f\n", rate, min, max, average);
  free(a);
  free(b);
  free(t);
  return 0;
}
