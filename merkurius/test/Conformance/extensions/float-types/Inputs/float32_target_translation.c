/* run-float32-target.m in C, sequentially, whose float is float32. */
#include <math.h>
#include <stdio.h>

int main(void) {
  enum { N = 1000 };
  float a[N], sum = 0.0f, max = -1.0f, b[4] = {0};
  for (int i = 0; i < N; ++i) {
    float x = (float)i;
    a[i] = sqrtf(x * x) * 0.5f + 0.25f;
  }
  for (int i = 0; i < N; ++i) {
    sum += a[i];
    max = max >= a[i] ? max : a[i];
  }
  for (int i = 0; i < N; ++i) {
    b[i % 4] += 0.25f;
  }
  printf("%g\n%.1f\n%g\n", a[10], sum, max);
  printf("array([%.1ff32, %.1ff32, %.1ff32, %.1ff32])\n", b[0], b[1], b[2],
      b[3]);
  return 0;
}
