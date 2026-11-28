// math_loop32.m's loops in C with OpenMP offload (README.md), in float.
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

static float run_sqrt(long n, long k) {
  float sum = 0.0f;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    float c = (float)i / (float)n, x = c;
    for (long j = 0; j < k; ++j)
      x = sqrtf(x + c);
    sum += x;
  }
  return sum;
}

static float run_exp(long n, long k) {
  float sum = 0.0f;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    float c = (float)i / (float)n, x = c;
    for (long j = 0; j < k; ++j)
      x = expf(-(x + c));
    sum += x;
  }
  return sum;
}

static float run_ln(long n, long k) {
  float sum = 0.0f;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    float c = (float)i / (float)n, x = c;
    for (long j = 0; j < k; ++j)
      x = logf(x + 2.0f + c);
    sum += x;
  }
  return sum;
}

static float run_sin(long n, long k) {
  float sum = 0.0f;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    float c = (float)i / (float)n, x = c;
    for (long j = 0; j < k; ++j)
      x = sinf(x + 1.0f + c);
    sum += x;
  }
  return sum;
}

static float run_cos(long n, long k) {
  float sum = 0.0f;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    float c = (float)i / (float)n, x = c;
    for (long j = 0; j < k; ++j)
      x = cosf(x + c);
    sum += x;
  }
  return sum;
}

static float run_pow(long n, long k) {
  float sum = 0.0f;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    float c = (float)i / (float)n, x = c;
    for (long j = 0; j < k; ++j)
      x = powf(x + 1.0f + c, 0.75f);
    sum += x;
  }
  return sum;
}

static float run_tanh(long n, long k) {
  float sum = 0.0f;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    float c = (float)i / (float)n, x = c;
    for (long j = 0; j < k; ++j)
      x = tanhf(x + 0.5f + c);
    sum += x;
  }
  return sum;
}

static void time_run(const char *name, float (*run)(long, long), long n,
                     long k, long repeat) {
  double first = run(n, k), best = 1.0e30;
  for (long r = 0; r < repeat; ++r) {
    double t0 = now();
    volatile float sink = run(n, k);
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
