// The memory bandwidth benchmark of bandwidth.m in C with OpenMP, to compare
// Merkurius with clang on the same machine: the same kernels, arrays, maps,
// timing, check and output, statement for statement where C allows it.
// See README.md.
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define KERNELS 5
#define START_A 0.1
#define START_B 0.2
#define START_C 0.0
#define SCALAR 0.4

// The kernels are separate functions, as in bandwidth.m, so that clang sees
// the arrays only through their parameters there too.
#define KERNEL __attribute__((noinline)) static

KERNEL void init(long n, double *a, double *b, double *c) {
#pragma omp target teams distribute parallel for map(tofrom : a[0 : n], b[0 : n], c[0 : n])
  for (long i = 0; i < n; i++) {
    a[i] = START_A;
    b[i] = START_B;
    c[i] = START_C;
  }
}

KERNEL void copy(long n, const double *a, double *c) {
#pragma omp target teams distribute parallel for map(to : a[0 : n]) map(tofrom : c[0 : n])
  for (long i = 0; i < n; i++)
    c[i] = a[i];
}

KERNEL void mul(long n, const double *c, double *b) {
#pragma omp target teams distribute parallel for map(to : c[0 : n]) map(tofrom : b[0 : n])
  for (long i = 0; i < n; i++)
    b[i] = SCALAR * c[i];
}

KERNEL void add(long n, const double *a, const double *b, double *c) {
#pragma omp target teams distribute parallel for map(to : a[0 : n], b[0 : n]) map(tofrom : c[0 : n])
  for (long i = 0; i < n; i++)
    c[i] = a[i] + b[i];
}

KERNEL void triad(long n, const double *b, const double *c, double *a) {
#pragma omp target teams distribute parallel for map(to : b[0 : n], c[0 : n]) map(tofrom : a[0 : n])
  for (long i = 0; i < n; i++)
    a[i] = b[i] + SCALAR * c[i];
}

KERNEL double dot(long n, const double *a, const double *b) {
  double sum = 0.0;
#pragma omp target teams distribute parallel for map(to : a[0 : n], b[0 : n]) map(tofrom : sum) reduction(+ : sum)
  for (long i = 0; i < n; i++)
    sum += a[i] * b[i];
  return sum;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static long setting(const char *name, long fallback) {
  const char *s = getenv(name);
  return s && *s ? strtol(s, NULL, 10) : fallback;
}

static double *zeroed(long n) {
  double *x = malloc((size_t)n * sizeof(double));
  if (!x) {
    printf("Out of memory.\n");
    exit(1);
  }
  for (long i = 0; i < n; i++)
    x[i] = 0.0;
  return x;
}

// The average of the elements' absolute errors.
static double error(long n, const double *x, double gold) {
  double sum = 0.0;
  for (long i = 0; i < n; i++)
    sum += fabs(x[i] - gold);
  return sum / (double)n;
}

static int check_array(const char *name, double err, double gold) {
  if (err / gold > 100.0 * DBL_EPSILON) {
    printf("Validation failed on %s[]. Average error %g\n", name, err);
    return 0;
  }
  return 1;
}

static int check(long n, long times, const double *a, const double *b,
                 const double *c, double sum) {
  double ga = START_A, gb = START_B, gc = START_C;
  for (long k = 0; k < times; k++) {
    gc = ga;
    gb = SCALAR * gc;
    gc = ga + gb;
    ga = gb + SCALAR * gc;
  }
  double gsum = ga * gb * (double)n;
  double esum = fabs((sum - gsum) / gsum);
  int valid = check_array("a", error(n, a, ga), ga);
  valid &= check_array("b", error(n, b, gb), gb);
  valid &= check_array("c", error(n, c, gc), gc);
  if (esum > 1.0e-8) {
    printf("Validation failed on sum. Error %g\nSum was %.17g but should be "
           "%.17g\n",
           (sum - gsum) / gsum, sum, gsum);
    valid = 0;
  }
  return valid;
}

// A kernel's row: its bandwidth from its fastest run, leaving out the first.
static void row(const char *name, int kernel, double bytes, long times,
                const double *t) {
  double min = DBL_MAX, max = 0.0, total = 0.0;
  for (long k = 1; k < times; k++) {
    double x = t[kernel * times + k];
    min = x < min ? x : min;
    max = x > max ? x : max;
    total += x;
  }
  printf("%-12s%-12.3f%-12.5f%-12.5f%-12.5f\n", name, 1.0e-6 * bytes / min, min,
         max, total / (double)(times - 1));
}

int main(void) {
  long n = setting("BANDWIDTH_ARRAY_SIZE", 33554432);
  long times = setting("BANDWIDTH_NUM_TIMES", 100);
  if (n < 1 || times < 2) {
    printf("BANDWIDTH_ARRAY_SIZE must be at least 1 and BANDWIDTH_NUM_TIMES at "
           "least 2.\n");
    return 1;
  }
  double mb = 1.0e-6 * 8.0 * (double)n;
  printf("C memory bandwidth benchmark\n");
  printf("Running kernels %ld times\n", times);
  printf("Precision: double\n");
  printf("Array size: %.1f MB (=%.1f GB)\n", mb, 1.0e-3 * mb);
  printf("Total size: %.1f MB (=%.1f GB)\n", 3.0 * mb, 3.0e-3 * mb);

  double *a = zeroed(n), *b = zeroed(n), *c = zeroed(n);
  init(n, a, b, c);
  double *t = zeroed(KERNELS * times);
  double sum = 0.0;
#pragma omp target data map(tofrom : a[0 : n], b[0 : n], c[0 : n])
  for (long k = 0; k < times; k++) {
    double t0 = now();
    copy(n, a, c);
    double t1 = now();
    mul(n, c, b);
    double t2 = now();
    add(n, a, b, c);
    double t3 = now();
    triad(n, b, c, a);
    double t4 = now();
    sum = dot(n, a, b);
    double t5 = now();
    t[0 * times + k] = t1 - t0;
    t[1 * times + k] = t2 - t1;
    t[2 * times + k] = t3 - t2;
    t[3 * times + k] = t4 - t3;
    t[4 * times + k] = t5 - t4;
  }
  if (!check(n, times, a, b, c, sum))
    return 1;
  printf("%-12s%-12s%-12s%-12s%-12s\n", "Function", "MBytes/sec", "Min (sec)",
         "Max", "Average");
  double size = (double)n * 8.0;
  row("Copy", 0, 2.0 * size, times, t);
  row("Mul", 1, 2.0 * size, times, t);
  row("Add", 2, 3.0 * size, times, t);
  row("Triad", 3, 3.0 * size, times, t);
  row("Dot", 4, 2.0 * size, times, t);
  return 0;
}
