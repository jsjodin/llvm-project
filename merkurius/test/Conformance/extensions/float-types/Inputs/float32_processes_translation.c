/* run-float32-processes.m in C, whose float is float32: the scaler's
   answers, each value printed as the Merkurius runtime prints it. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *w32(float x) {
  static char buffer[8][64];
  static int next;
  char *b = buffer[next++ % 8];
  for (int p = 6; p <= 9; ++p) {
    snprintf(b, 64, "%.*g", p, (double)x);
    if (strtof(b, NULL) == x) {
      break;
    }
  }
  if (!strpbrk(b, ".e")) {
    strcat(b, ".0");
  }
  strcat(b, "f32");
  return b;
}

int main(void) {
  printf("message(half(%s))\n", w32(0.1f / 2.0f));
  float a[4] = {1.0f, 2.5f, -3.0f, 1.0e-3f}, s = 0.0f;
  for (int i = 0; i < 4; ++i) {
    a[i] = 0.5f * a[i];
    s = s + a[i];
  }
  printf("message(scaled(%s, array([%s, %s, %s, %s])))\n", w32(s), w32(a[0]),
      w32(a[1]), w32(a[2]), w32(a[3]));
  printf("message(length(%s, 7))\n", w32(sqrtf(3.0f * 3.0f + 4.0f * 4.0f)));
  printf("message(length(%s, 8))\n", w32(sqrtf(1.0f * 1.0f + 1.0f * 1.0f)));
  return 0;
}
