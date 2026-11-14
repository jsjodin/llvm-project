// pipeline.m's three stages as HIP kernels, one after another in one stream,
// the frames in turn: the bar for the pipeline of GPU processes
// (docs/INPROGRESS.md, A pipeline of processes). The same frames, work,
// feedback and checksum as pipeline.m, so the same bits. Built with
// -ffp-contract=off, as merkc fuses no multiply and add.
//
// With PIPE_FEEDBACK=D above 0, the kernel that makes frame F reads frame
// F - D's checksum from the GPU's memory, where the stream's order has put
// it: feedback costs nothing here.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

enum { THREADS = 256 };

static long setting(const char *name, long fallback) {
  const char *s = getenv(name);
  return s && *s ? strtol(s, NULL, 10) : fallback;
}

static void check(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    printf("%s: %s\n", what, hipGetErrorString(e));
    exit(1);
  }
}

__device__ static double rounds(long work, double a, double b, double x) {
  for (long k = 0; k < work; k++)
    x = x * a + b;
  return x;
}

// Frame f's cells, its feedback mixed in, after the first stage's work.
__global__ static void make(long n, long work, long f, long feedback,
    const uint64_t *sums, double *out) {
  long mix =
      feedback > 0 && f >= feedback ? (long)(sums[f - feedback] & 1023) : 0;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n;
      i += (long)gridDim.x * blockDim.x)
    out[i] = rounds(
        work, 0.9999999, 0.0000001, (double)((f * n + i + mix) & 1023) * 0.001);
}

__global__ static void stage(
    long n, long work, double a, double b, const double *in, double *out) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n;
      i += (long)gridDim.x * blockDim.x)
    out[i] = rounds(work, a, b, in[i]);
}

// The third stage, and frame f's checksum into sums[f].
__global__ static void last(
    long n, long work, long f, const double *in, double *out, uint64_t *sums) {
  __shared__ uint64_t part[THREADS];
  uint64_t sum = 0;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n;
      i += (long)gridDim.x * blockDim.x) {
    double x = rounds(work, 0.9999997, 0.0000002, in[i]);
    out[i] = x;
    double y = x + 0.0;
    uint64_t bits;
    memcpy(&bits, &y, sizeof bits);
    sum += bits * (2 * (uint64_t)i + 1);
  }
  part[threadIdx.x] = sum;
  __syncthreads();
  for (int half = THREADS / 2; half > 0; half /= 2) {
    if (threadIdx.x < half)
      part[threadIdx.x] += part[threadIdx.x + half];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd((unsigned long long *)&sums[f], (unsigned long long)part[0]);
}

int main(void) {
  long n = setting("PIPE_WORDS", 65536);
  long work = setting("PIPE_WORK", 64);
  long frames = setting("PIPE_FRAMES", 200);
  long feedback = setting("PIPE_FEEDBACK", 0);
  if (n < 1 || work < 0 || frames < 2 || feedback < 0) {
    printf("PIPE_WORDS must be at least 1, PIPE_WORK and PIPE_FEEDBACK at "
           "least 0, and PIPE_FRAMES at least 2.\n");
    return 1;
  }
  printf("Pipeline, HIP, kernels in one stream\n");
  printf("Frames: %ld of %ld words, %ld rounds a stage\n", frames, n, work);
  if (feedback > 0)
    printf("Feedback: %ld frames in flight\n", feedback);
  else
    printf("Feedback: none\n");
  int units = 0;
  check(hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, 0),
      "hipDeviceGetAttribute");
  long blocks = (n + THREADS - 1) / THREADS;
  // Enough blocks for every unit the mask leaves, and no more than cells.
  long most = 16L * units;
  blocks = blocks < most ? blocks : most;
  double *a, *b, *c;
  uint64_t *sums;
  check(hipMalloc(&a, n * sizeof(double)), "hipMalloc");
  check(hipMalloc(&b, n * sizeof(double)), "hipMalloc");
  check(hipMalloc(&c, n * sizeof(double)), "hipMalloc");
  check(hipMalloc(&sums, frames * sizeof(uint64_t)), "hipMalloc");
  check(hipMemset(sums, 0, frames * sizeof(uint64_t)), "hipMemset");
  hipEvent_t first, end;
  check(hipEventCreate(&first), "hipEventCreate");
  check(hipEventCreate(&end), "hipEventCreate");
  for (long f = 0; f < frames; f++) {
    make<<<blocks, THREADS>>>(n, work, f, feedback, sums, a);
    stage<<<blocks, THREADS>>>(n, work, 1.0000001, -0.0000003, a, b);
    last<<<blocks, THREADS>>>(n, work, f, b, c, sums);
    if (f == 0)
      check(hipEventRecord(first, 0), "hipEventRecord");
  }
  check(hipEventRecord(end, 0), "hipEventRecord");
  check(hipEventSynchronize(end), "hipEventSynchronize");
  float ms = 0.0f;
  check(hipEventElapsedTime(&ms, first, end), "hipEventElapsedTime");
  uint64_t *host = (uint64_t *)malloc(frames * sizeof(uint64_t));
  check(hipMemcpy(host, sums, frames * sizeof(uint64_t), hipMemcpyDeviceToHost),
      "hipMemcpy");
  uint64_t checksum = 0;
  for (long f = 0; f < frames; f++)
    checksum += host[f];
  printf("Checksum: %016llx\n", (unsigned long long)checksum);
  printf("Time a frame: %.1f us\n", 1000.0 * ms / (double)(frames - 1));
  free(host);
  check(hipFree(a), "hipFree");
  check(hipFree(b), "hipFree");
  check(hipFree(c), "hipFree");
  check(hipFree(sums), "hipFree");
  return 0;
}
