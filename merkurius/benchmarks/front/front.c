// The reacting front of front.m in C with OpenMP, the bar that Merkurius is
// measured against: the same grid, start, model, kernels, maps, timing,
// check and output, statement for statement where C allows it. See
// README.md.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// The model (README.md): diffusion of heat, a loss of heat to the
// surroundings, and two reactions, fuel to an intermediate and the
// intermediate to product, each giving heat.
#define ALPHA 0.2
#define LOSS 0.01
#define IGNITE1 1.5
#define IGNITE2 2.0
#define RATE1 0.5
#define RATE2 2.0
#define HEAT1 1.0
#define HEAT2 3.0
#define GONE 1.0e-9

// The kernels are separate functions, as in front.m, so that clang sees the
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

// Cold and full of fuel, but for a hot disc of radius 5 at a quarter of the
// way along each side; the border is cold, and stays so.
KERNEL void init(long n, double *t, double *u, double *y, double *z) {
  long w = n + 2;
#pragma omp target teams distribute parallel for                              \
    map(tofrom : t[0 : w * w], u[0 : w * w], y[0 : w * w], z[0 : w * w])
  for (long k = 0; k < w * w; k++) {
    long r = k / w - 1, c = k % w - 1;
    double dr = (double)r - (double)n / 4.0, dc = (double)c - (double)n / 4.0;
    int inside = r >= 0 && r < n && c >= 0 && c < n;
    t[k] = inside && dr * dr + dc * dc < 25.0 ? 5.0 : 1.0;
    u[k] = 1.0;
    y[k] = inside ? 1.0 : 0.0;
    z[k] = 0.0;
  }
}

// One step: u's temperatures from t's, and each cell's fuel and
// intermediate in place, in `substeps` substeps where the cell burns.
KERNEL void step(long n, long substeps, const double *t, double *u, double *y,
                 double *z) {
  long w = n + 2;
  double h = 1.0 / (double)substeps;
#pragma omp target teams distribute parallel for map(to : t[0 : w * w])      \
    map(tofrom : u[0 : w * w], y[0 : w * w], z[0 : w * w])
  for (long k = 0; k < n * n; k++) {
    long c = (k / n + 1) * w + k % n + 1;
    double heat = t[c] + ALPHA * (t[c - w] + t[c + w] + t[c - 1] + t[c + 1] -
                                  4.0 * t[c]);
    heat = heat - LOSS * (heat - 1.0);
    double fuel = y[c], middle = z[c];
    if (heat > IGNITE1 && fuel + middle > 0.0) {
      for (long j = 0; j < substeps; j++) {
        double k1 = heat > IGNITE1
                        ? RATE1 * (heat - IGNITE1) * (heat - IGNITE1)
                        : 0.0;
        double fuel1 = fuel / (1.0 + h * k1);
        double burnt1 = fuel - fuel1;
        fuel = fuel1;
        middle = middle + burnt1;
        heat = heat + HEAT1 * burnt1;
        double k2 = heat > IGNITE2 ? RATE2 * (heat - IGNITE2) : 0.0;
        double middle1 = middle / (1.0 + h * k2);
        double burnt2 = middle - middle1;
        middle = middle1;
        heat = heat + HEAT2 * burnt2;
      }
      fuel = fuel < GONE ? 0.0 : fuel;
      middle = middle < GONE ? 0.0 : middle;
    }
    u[c] = heat;
    y[c] = fuel;
    z[c] = middle;
  }
}

int main(void) {
  long n = setting("FRONT_SIZE", 4096);
  long steps = setting("FRONT_STEPS", 100);
  long substeps = setting("FRONT_SUBSTEPS", 100);
  if (n < 1 || steps < 2 || substeps < 1) {
    printf("FRONT_SIZE and FRONT_SUBSTEPS must be at least 1 and FRONT_STEPS "
           "at least 2.\n");
    return 1;
  }
  printf("Reacting front, C\n");
  printf("Grid: %ld x %ld cells\n", n, n);
  printf("Steps: %ld, of %ld substeps where a cell burns\n", steps, substeps);
  long w = n + 2;
  double *t = calloc(w * w, sizeof(double));
  double *u = calloc(w * w, sizeof(double));
  double *y = calloc(w * w, sizeof(double));
  double *z = calloc(w * w, sizeof(double));
  double *times = calloc(steps, sizeof(double));
  if (!t || !u || !y || !z || !times) {
    printf("out of memory\n");
    return 1;
  }
  init(n, t, u, y, z);
  // The arrays stay on the device for the steps; each step's own maps find
  // them there and copy nothing.
  double *final = t;
#pragma omp target data map(tofrom : t[0 : w * w], u[0 : w * w],              \
                                y[0 : w * w], z[0 : w * w])
  {
    double *cur = t, *next = u;
    for (long s = 0; s < steps; s++) {
      double t0 = now();
      step(n, substeps, cur, next, y, z);
      double t1 = now();
      times[s] = t1 - t0;
      double *swap = cur;
      cur = next;
      next = swap;
    }
    final = cur;
  }
  // Cells by rows, in order, so that every build adds the same way.
  long burnt = 0;
  double heat = 0.0, fuel = 0.0;
  for (long k = 0; k < n * n; k++) {
    long c = (k / n + 1) * w + k % n + 1;
    burnt += y[c] < 0.5 ? 1 : 0;
    heat += final[c];
    fuel += y[c];
  }
  // Each step but the first, which includes starting the kernels.
  double min = times[1], max = times[1], sum = 0.0;
  for (long s = 1; s < steps; s++) {
    min = times[s] < min ? times[s] : min;
    max = times[s] > max ? times[s] : max;
    sum += times[s];
  }
  double average = sum / (double)(steps - 1);
  double rate = (double)n * (double)n / min * 1.0e-9;
  printf("Burnt cells: %ld\n", burnt);
  printf("Heat: %.15e\n", heat);
  printf("Fuel: %.15e\n", fuel);
  printf("%-14s%-12s%-12s%-12s\n", "Gcells/sec", "Min (sec)", "Max",
         "Average");
  printf("%-14.3f%-12.6f%-12.6f%-12.6f\n", rate, min, max, average);
  free(t);
  free(u);
  free(y);
  free(z);
  free(times);
  return 0;
}
