// What swe-sparse.c and swe-sparse.hip share, so that their cells are
// computed by the same code: the scheme, the cases, and the host's tables of
// the tiles and their bookkeeping, as life-sparse.c's. swe-sparse.m has the
// same operations in the same order (docs/INPROGRESS.md, A showcase with
// more work per tile). A program defines SWE_DEVICE before including it,
// as what makes a function the device's too, and may define SWE_CONST.
#ifndef SWE_H
#define SWE_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

#define SWE_FN SWE_DEVICE static inline
// How the scheme's constants are declared: C's static const, which OpenMP's
// declare target makes the GPU's; HIP's static constexpr.
#ifndef SWE_CONST
#define SWE_CONST static const
#endif

enum { QUIET_STEPS = 4, ALL_EDGES = 0x1ef, ENDS = 0x200 };

// Gravity, the depth of a dry cell, the bowl's depth at its centre, the
// column's depth above the bed, the Courant number, Thacker's amplitude.
SWE_CONST double G = 9.81, DRY = 1.0e-6, D0 = 10.0, COLUMN = 5.0,
                 COURANT = 0.45, A = 0.4;

// Mercury's float.max and float.min.
SWE_FN double maxf(double x, double y) { return x >= y ? x : y; }
SWE_FN double minf(double x, double y) { return x <= y ? x : y; }

// The bed at the centre of the cell at row y and column x of a world of
// side n, a bowl of radius 0.45 n.
SWE_FN double bed(int64_t y, int64_t x, int64_t n) {
  double c = 0.5 * (double)n, r = 0.45 * (double)n;
  double dx = ((double)x + 0.5) - c, dy = ((double)y + 0.5) - c;
  return D0 * ((dx * dx + dy * dy) / (r * r) - 1.0);
}

// Thacker's surface at the cell's centre at time t, and the depth there.
SWE_FN double thacker(int64_t y, int64_t x, int64_t n, double t) {
  double c = 0.5 * (double)n, r = 0.45 * (double)n;
  double dx = ((double)x + 0.5) - c, dy = ((double)y + 0.5) - c;
  double w = sqrt(8.0 * G * D0) / r;
  double q = 1.0 - A * cos(w * t);
  double s = D0 *
      (sqrt(1.0 - A * A) / q - 1.0 -
          (dx * dx + dy * dy) / (r * r) * ((1.0 - A * A) / (q * q) - 1.0));
  return maxf(0.0, s - bed(y, x, n));
}

// The cell's depth at the start.
SWE_FN double start_depth(int64_t y, int64_t x, int64_t n, int kase) {
  if (kase == 1)
    return thacker(y, x, n, 0.0);
  double c = 0.5 * (double)n, cx = c + 0.2 * (double)n;
  double rc = (double)n / 16.0;
  double dx = ((double)x + 0.5) - cx, dy = ((double)y + 0.5) - c;
  if (dx * dx + dy * dy > rc * rc)
    return 0.0;
  // A flat top COLUMN above the bed under the column's centre.
  double top = D0 *
          ((0.2 * (double)n) * (0.2 * (double)n) /
                  ((0.45 * (double)n) * (0.45 * (double)n)) -
              1.0) +
      COLUMN;
  return maxf(0.0, top - bed(y, x, n));
}

// The flux through the face between a left and a right cell, each its
// depth, its momenta normal to the face and along it, and its bed: the
// mass, normal and along fluxes, and the left's and the right's pressure
// corrections of hydrostatic reconstruction.
struct flux {
  double m, n, a, pl, pr;
};

SWE_FN struct flux hll(double hl, double nl, double al, double bl, double hr,
    double nr, double ar, double br) {
  double ul = hl > DRY ? nl / hl : 0.0, vl = hl > DRY ? al / hl : 0.0;
  double ur = hr > DRY ? nr / hr : 0.0, vr = hr > DRY ? ar / hr : 0.0;
  double bf = maxf(bl, br);
  double hls = maxf(0.0, hl + bl - bf), hrs = maxf(0.0, hr + br - bf);
  double cl = sqrt(G * hls), cr = sqrt(G * hrs);
  double sl = minf(ul - cl, ur - cr), sr = maxf(ul + cl, ur + cr);
  double fml = hls * ul, fnl = hls * ul * ul + 0.5 * G * hls * hls,
         fal = hls * ul * vl;
  double fmr = hrs * ur, fnr = hrs * ur * ur + 0.5 * G * hrs * hrs,
         far = hrs * ur * vr;
  struct flux f;
  if (sl >= 0.0) {
    f.m = fml;
    f.n = fnl;
    f.a = fal;
  } else if (sr <= 0.0) {
    f.m = fmr;
    f.n = fnr;
    f.a = far;
  } else {
    double d = sr - sl;
    f.m = (sr * fml - sl * fmr + sl * sr * (hrs - hls)) / d;
    f.n = (sr * fnl - sl * fnr + sl * sr * (hrs * ur - hls * ul)) / d;
    f.a = (sr * fal - sl * far + sl * sr * (hrs * vr - hls * vl)) / d;
  }
  f.pl = 0.5 * G * (hl * hl - hls * hls);
  f.pr = 0.5 * G * (hr * hr - hrs * hrs);
  return f;
}

// The directions that a wet cell at row r and column c of a tile whose
// last row and column are m makes wet: its edge, if any, each, and the
// corner.
SWE_FN int edge_bits(int64_t r, int64_t c, int64_t m) {
  int dy = r == 0 ? -1 : r == m ? 1 : 0;
  int dx = c == 0 ? -1 : c == m ? 1 : 0;
  int bits = 0;
  if (dy != 0)
    bits |= 1 << ((dy + 1) * 3 + 1);
  if (dx != 0)
    bits |= 1 << (3 + dx + 1);
  if (dy != 0 && dx != 0)
    bits |= 1 << ((dy + 1) * 3 + dx + 1);
  return bits;
}

// The cell's wave speed, |u| + |v| + 2 sqrt(g h), 0 if it is dry.
SWE_FN double speed(double h, double hu, double hv) {
  if (h <= DRY)
    return 0.0;
  double c = sqrt(G * h);
  return fabs(hu / h) + fabs(hv / h) + 2.0 * c;
}

// A double's bits, -0.0 as 0.0.
SWE_FN uint64_t bits(double x) {
  double y = x + 0.0;
  uint64_t b;
  memcpy(&b, &y, sizeof b);
  return b;
}

// The world's shape and the host's tables of the tiles.
struct world {
  int64_t tile, log, side, size, capacity;
  int32_t *dir; // A tile for each place, or 0.
  int32_t *pos; // Each tile's place.
  int32_t *nb; // Each tile's nine neighbours, by tile.
  int32_t *wet; // Each tile's wet cells after the last step.
  uint16_t *edges; // Each tile's wet edges after the last step.
  uint8_t *quiet; // The steps each tile has been quiet.
  int32_t *actives; // The active tiles, nactive of them.
  int32_t *free; // The free tiles, nfree of them.
  uint16_t *fate; // Each active tile's fate, by its slot.
  int64_t nactive, nfree;
  int overflow;
  // What the GPU reads and writes, by the active tiles' slots: their
  // neighbours, and the step's results.
  int32_t *slot_nb;
  int32_t *slot_wet;
  uint16_t *slot_edges;
  double *slot_speed;
};

// The place in direction d of place p, or -1 if it is outside the world.
static int64_t beside(const struct world *w, int64_t p, int d) {
  int64_t y = p / w->side + d / 3 - 1, x = p % w->side + d % 3 - 1;
  if (y < 0 || y >= w->side || x < 0 || x >= w->side)
    return -1;
  return y * w->side + x;
}

// A tile at the place, from the free list, linked to its neighbours both
// ways and active; or the overflow set if none is free.
static void start(struct world *w, int64_t place) {
  if (w->nfree == 0) {
    w->overflow = 1;
    return;
  }
  int32_t i = w->free[--w->nfree];
  w->dir[place] = i;
  w->pos[i] = (int32_t)place;
  w->quiet[i] = 0;
  for (int d = 0; d < 9; d++) {
    int64_t there = beside(w, place, d);
    int32_t n = d == 4 ? i : there < 0 ? 0 : w->dir[there];
    w->nb[i * 9 + d] = n;
    if (d != 4 && n != 0)
      w->nb[n * 9 + 8 - d] = i;
  }
  w->actives[w->nactive++] = i;
}

// Each active tile's fate after a step: its quiet steps, which end it at
// QUIET_STEPS, and the directions where it starts a tile. Whether any tile
// has one.
static int decide(struct world *w) {
  int pending = 0;
  for (int64_t j = 0; j < w->nactive; j++) {
    int32_t i = w->actives[j];
    const int32_t *n = w->nb + i * 9;
    int missing = 0, facing = 0;
    for (int d = 0; d < 9; d++) {
      if (d == 4)
        continue;
      if (n[d] == 0)
        missing |= 1 << d;
      else
        facing |= w->edges[n[d]] & (1 << (8 - d));
    }
    int inside = 0;
    for (int d = 0; d < 9; d++)
      if (beside(w, w->pos[i], d) >= 0)
        inside |= 1 << d;
    int fate = w->edges[i] & missing & inside;
    int q = w->wet[i] == 0 && facing == 0 ? w->quiet[i] + 1 : 0;
    w->quiet[i] = (uint8_t)q;
    if (q >= QUIET_STEPS)
      fate |= ENDS;
    w->fate[j] = (uint16_t)fate;
    pending |= fate != 0;
  }
  return pending;
}

// The tiles whose fates end them end, each slot taken by the last one's.
static void finish(struct world *w) {
  for (int64_t j = 0; j < w->nactive;) {
    if (!(w->fate[j] & ENDS)) {
      j++;
      continue;
    }
    int32_t i = w->actives[j];
    for (int d = 0; d < 9; d++) {
      int32_t n = w->nb[i * 9 + d];
      if (d != 4 && n != 0)
        w->nb[n * 9 + 8 - d] = 0;
    }
    w->dir[w->pos[i]] = 0;
    w->free[w->nfree++] = i;
    w->nactive--;
    w->actives[j] = w->actives[w->nactive];
    w->fate[j] = w->fate[w->nactive];
  }
}

// The tiles the fates of the first kept active tiles ask for, where none
// has started since.
static void grow(struct world *w, int64_t kept) {
  for (int64_t j = 0; j < kept; j++) {
    int bits = w->fate[j] & 0x1ff;
    int32_t i = w->actives[j];
    for (int d = 0; d < 9 && bits != 0; d++) {
      if (!(bits & (1 << d)) || w->nb[i * 9 + d] != 0)
        continue;
      int64_t place = beside(w, w->pos[i], d);
      if (place >= 0)
        start(w, place);
    }
  }
}

// The active tiles' neighbours by their slots, for the GPU.
static void slots(struct world *w) {
  for (int64_t j = 0; j < w->nactive; j++)
    memcpy(w->slot_nb + j * 9, w->nb + w->actives[j] * 9, 9 * sizeof(int32_t));
}

#endif
