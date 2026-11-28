// math_loop32.m's loops in HIP (README.md), in float: a thread an element,
// a block's sum by atomics, so that the reduction costs little against the
// loop.
#include <hip/hip_runtime.h>
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

__global__ void kernel_sqrt(long n, long k, float *sum) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float c = (float)i / (float)n, x = c;
  for (long j = 0; j < k; ++j)
    x = sqrtf(x + c);
  atomicAdd(sum, x);
}

__global__ void kernel_exp(long n, long k, float *sum) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float c = (float)i / (float)n, x = c;
  for (long j = 0; j < k; ++j)
    x = expf(-(x + c));
  atomicAdd(sum, x);
}

__global__ void kernel_ln(long n, long k, float *sum) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float c = (float)i / (float)n, x = c;
  for (long j = 0; j < k; ++j)
    x = logf(x + 2.0f + c);
  atomicAdd(sum, x);
}

__global__ void kernel_sin(long n, long k, float *sum) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float c = (float)i / (float)n, x = c;
  for (long j = 0; j < k; ++j)
    x = sinf(x + 1.0f + c);
  atomicAdd(sum, x);
}

__global__ void kernel_cos(long n, long k, float *sum) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float c = (float)i / (float)n, x = c;
  for (long j = 0; j < k; ++j)
    x = cosf(x + c);
  atomicAdd(sum, x);
}

__global__ void kernel_pow(long n, long k, float *sum) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float c = (float)i / (float)n, x = c;
  for (long j = 0; j < k; ++j)
    x = powf(x + 1.0f + c, 0.75f);
  atomicAdd(sum, x);
}

__global__ void kernel_tanh(long n, long k, float *sum) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float c = (float)i / (float)n, x = c;
  for (long j = 0; j < k; ++j)
    x = tanhf(x + 0.5f + c);
  atomicAdd(sum, x);
}

typedef void (*Kernel)(long, long, float *);

static float run(Kernel kernel, long n, long k, float *sum) {
  float host = 0.0f;
  (void)hipMemcpy(sum, &host, sizeof host, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(kernel, dim3((n + 255) / 256), dim3(256), 0, 0, n, k, sum);
  (void)hipMemcpy(&host, sum, sizeof host, hipMemcpyDeviceToHost);
  return host;
}

static void time_run(const char *name, Kernel kernel, long n, long k,
                     long repeat, float *sum) {
  double first = run(kernel, n, k, sum), best = 1.0e30;
  for (long r = 0; r < repeat; ++r) {
    double t0 = now();
    volatile float sink = run(kernel, n, k, sum);
    (void)sink;
    double t = now() - t0;
    best = t < best ? t : best;
  }
  printf("%-6s %10.6f %.17g\n", name, best, first);
}

int main(void) {
  long n = setting("MATH_N", 1L << 20), k = setting("MATH_K", 1000),
       repeat = setting("MATH_REPEAT", 5);
  float *sum;
  (void)hipMalloc(&sum, sizeof *sum);
  time_run("sqrt", kernel_sqrt, n, k, repeat, sum);
  time_run("exp", kernel_exp, n, k, repeat, sum);
  time_run("ln", kernel_ln, n, k, repeat, sum);
  time_run("sin", kernel_sin, n, k, repeat, sum);
  time_run("cos", kernel_cos, n, k, repeat, sum);
  time_run("pow", kernel_pow, n, k, repeat, sum);
  time_run("tanh", kernel_tanh, n, k, repeat, sum);
  (void)hipFree(sum);
  return 0;
}
