/* run-float32.m in C, whose float is float32: each value printed as the
   Merkurius runtime prints it (runtime/write.cpp). */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *f32(float x) {
  static char buffer[8][64];
  static int next;
  char *b = buffer[next++ % 8];
  if (isnan(x)) {
    strcpy(b, "nan");
  } else if (isinf(x)) {
    strcpy(b, x < 0 ? "-infinity" : "infinity");
  } else {
    for (int p = 6; p <= 9; ++p) {
      snprintf(b, 64, "%.*g", p, (double)x);
      if (strtof(b, NULL) == x) {
        break;
      }
    }
    if (!strpbrk(b, ".e")) {
      strcat(b, ".0");
    }
  }
  return b;
}

/* As io.write writes a float32: with its suffix, if finite. */
static const char *w32(float x) {
  static char buffer[8][64];
  static int next;
  char *b = buffer[next++ % 8];
  snprintf(b, 64, "%s%s", f32(x), isfinite(x) ? "f32" : "");
  return b;
}

static const char *f64(double x) {
  static char b[64];
  for (int p = 15; p <= 17; ++p) {
    snprintf(b, 64, "%.*g", p, x);
    if (strtod(b, NULL) == x) {
      break;
    }
  }
  if (!strpbrk(b, ".e")) {
    strcat(b, ".0");
  }
  return b;
}

static const char *name(float x) {
  return x == 0.5f ? "half" : x == 0.1f ? "tenth" : "other";
}

int main(void) {
  float a = 1.5f, b = 0.1f;
  printf("%s\n", f32(a + b));
  printf("%s\n", f64((double)b));
  printf("%s\n", f32((float)0.1));
  printf("%s\n", f32(a / 3.0f));
  printf("%s\n", f32(1.0f / 3.0f));
  printf("[%s, %s, %s, %s, %s]\n", w32(a), w32(b), w32(-2.5f), w32(1.0e10f),
      w32((float)1.0e39));
  printf("[%s, %s]\n", w32(2.0f * a), w32(2.0f * b));
  printf("[\"%s\", \"%s\", \"%s\"]\n", name(0.5f), name(0.1f), name(2.0f));
  printf("{0.5, %s}\n", w32(0.5f));
  printf("point(%s, 7, %s)\n", w32(1.0f), w32(2.0e10f));
  printf("%s\n", w32(2.0e10f - 1.0f));
  float arr[4] = {0.25f, 0.25f, 7.5f, 0.25f};
  printf("array([%s, %s, %s, %s])\n", w32(arr[0]), w32(arr[1]), w32(arr[2]),
      w32(arr[3]));
  float s = 0.0f;
  for (int i = 0; i < 4; ++i) {
    s = arr[i] + s;
  }
  printf("%s\n", w32(s));
  printf("array([point(%s, 1, %s), point(%s, 2, %s)])\n", w32(1.0f), w32(2.0f),
      w32(3.0f), w32(4.0f));
  printf(a > b ? "gt\n" : "le\n");
  printf(0.0f == -0.0f ? "zeros equal\n" : "zeros differ\n");
  printf("%s\n", b < a ? "'<'" : b > a ? "'>'" : "'='");
  printf("%d\n", (int)truncf(2.75f));
  printf("%d\n", (int)floorf(-2.5f + 0.5f));
  printf("%s\n", f32((float)16777217));
  printf("%s\n", f32(FLT_MAX));
  printf("%s\n", f32(FLT_MIN));
  printf("%s\n", f32(FLT_EPSILON));
  printf("%s\n", f32((float)1.0e300));
  printf("%s\n", f32(1.5f * 1.5f * 1.5f));
  printf("%s\n", f32(sqrtf(2.0f)));
  printf("%s\n", f64(sqrt(2.0)));
  printf("%s\n", f32(3.14159265358979323846f));
  printf("%s\n", f32(expf(1.0f)));
  printf("%s\n", f32(atan2f(1.0f, 1.0f) * 4.0f));
  printf("%s\n", f32(logf(8.0f) / logf(2.0f)));
  printf("{%s, %s, 1.0, %s}\n", w32(1e+10f), w32(2.5e-3f), w32(3.0E2f));
  return 0;
}
