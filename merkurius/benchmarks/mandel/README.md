# A deep Mandelbrot zoom

The stepping stone to moving tiles between places (`docs/INPROGRESS.md`,
step 4 of the demo of a stencil whose active region moves): a zoom into the
Mandelbrot set's boundary, where a tile's cost varies by orders of magnitude,
unpredictably, from frame to frame, so that stealing beats a split made in
advance. No tile starts or ends, so balancing is tested alone.

## The zoom

- Frame f is `MANDEL_SIZE` pixels square (2048), of width `3 * 0.5^f`,
  centred on (-0.743643887037151, 0.131825904205330), for `MANDEL_FRAMES`
  frames (40). The last frame is 5.5e-12 wide, where a pixel's step is some
  25 units in the last place of the centre's coordinates; much deeper, the
  pixels of a 2048 frame would collapse onto each other in doubles.
- Pixel (px, py) is c = (X0 + float(px - Size / 2) * Step, Y0 + float(py -
  Size / 2) * Step), with Step = Width / float(Size).
- Its iterations, at most frame f's escape limit `256 + 64 * f`, follow one
  order of operations: `X2 = X * X`, `Y2 = Y * Y`, escape if `X2 + Y2 >
  4.0`, else `Y := (2.0 * X) * Y + Cy` and `X := X2 - Y2 + Cx`. merkc fuses
  no multiply and add, and C is built with `-ffp-contract=off`, so every
  program gives the same counts.
- The output: the iterations of the whole zoom, a checksum that weighs frame
  f's by f + 1, and the time from the first frame's start to the last's end,
  with the rate; `MANDEL_TRACE=1` prints each frame's limit, iterations and
  time.

## Programs

- `mandel.c`, the C reference: a kernel a frame, `target teams distribute
  parallel for` over its pixels, with a reduction of their iterations. The
  GPU's hardware balances the frame's work groups across its compute units.
- `mandel.m`: the same in one GPU process, a loop under `distribute
  parallel do` over a frame's pixels across all its worker teams, with a
  reduction. Its time is the host's, from before the spawn to the last
  answer, as `mandel.c`'s covers its kernels' launches. The first argument
  `host` spawns the process at the host.
- `mandel.py`, independent, in Python, whose floats are doubles that it does
  not fuse either: the counts for small frames, which the tests check the
  programs against.

`compare-c.sh` runs `mandel.m` and `mandel.c` in turns, checks that they give
the same iterations, and prints each one's median time and rate.

2026-11-12, gfx1030, on compute units 0-39, 40 frames, the median of 5
rounds of the whole zoom's time:

| Frames | Mercury, GPU process | C, a kernel a frame |
| --- | ---: | ---: |
| 2048² | 3.45 s, 38.8 Giterations/s | 3.57 s, 37.4 Giterations/s |
| 1024² | 0.90 s, 37.0 Giterations/s | 0.98 s, 34.0 Giterations/s |

## Tests

`test/Benchmarks/mandel.test` runs `mandel.m` at the host, sequentially at
-O0 and -O2 and on the host's threads, and `mandel.c` on the host's threads,
against `mandel.py` on 64 pixels for 12 frames and 51 for 30;
`mandel-gpu.test` both on the GPU, the whole zoom at 512 pixels too.
