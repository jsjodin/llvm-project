// math_loop.m's loops in C with OpenMP offload (README.md).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static long setting(const char *name, long fallback) {
  const char *s = getenv(name);
  return s && *s ? strtol(s, NULL, 10) : fallback;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static double run_sqrt(long n, long k) {
  double sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    double c = (double)i / (double)n, x = c;
    for (long j = 0; j < k; ++j)
      x = sqrt(x + c);
    sum += x;
  }
  return sum;
}

static double run_exp(long n, long k) {
  double sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    double c = (double)i / (double)n, x = c;
    for (long j = 0; j < k; ++j)
      x = exp(-(x + c));
    sum += x;
  }
  return sum;
}

static double run_ln(long n, long k) {
  double sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    double c = (double)i / (double)n, x = c;
    for (long j = 0; j < k; ++j)
      x = log(x + 2.0 + c);
    sum += x;
  }
  return sum;
}

static double run_sin(long n, long k) {
  double sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    double c = (double)i / (double)n, x = c;
    for (long j = 0; j < k; ++j)
      x = sin(x + 1.0 + c);
    sum += x;
  }
  return sum;
}

static double run_cos(long n, long k) {
  double sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    double c = (double)i / (double)n, x = c;
    for (long j = 0; j < k; ++j)
      x = cos(x + c);
    sum += x;
  }
  return sum;
}

static double run_pow(long n, long k) {
  double sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    double c = (double)i / (double)n, x = c;
    for (long j = 0; j < k; ++j)
      x = pow(x + 1.0 + c, 0.75);
    sum += x;
  }
  return sum;
}

static double run_tanh(long n, long k) {
  double sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    double c = (double)i / (double)n, x = c;
    for (long j = 0; j < k; ++j)
      x = tanh(x + 0.5 + c);
    sum += x;
  }
  return sum;
}

static void time_run(const char *name, double (*run)(long, long), long n,
                     long k, long repeat) {
  double first = run(n, k), best = 1.0e30;
  for (long r = 0; r < repeat; ++r) {
    double t0 = now();
    volatile double sink = run(n, k);
    (void)sink;
    double t = now() - t0;
    best = t < best ? t : best;
  }
  printf("%-6s %10.6f %.17g\n", name, best, first);
}

int main(void) {
  long n = setting("MATH_N", 1L << 20), k = setting("MATH_K", 1000),
       repeat = setting("MATH_REPEAT", 5);
  time_run("sqrt", run_sqrt, n, k, repeat);
  time_run("exp", run_exp, n, k, repeat);
  time_run("ln", run_ln, n, k, repeat);
  time_run("sin", run_sin, n, k, repeat);
  time_run("cos", run_cos, n, k, repeat);
  time_run("pow", run_pow, n, k, repeat);
  time_run("tanh", run_tanh, n, k, repeat);
  return 0;
}
