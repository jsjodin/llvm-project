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
- `mandel-places.m`: the zoom on `MANDEL_PLACES` places (4), GPU processes
  on one GPU with a `MANDEL_PLACES`'th of the kernel's teams each
  (`process.set_worker_teams`), and the host as the coordinator. A frame is
  in tiles of `MANDEL_TILE` pixels square (128, a power of two that divides
  the size). A place steps one tile at a time, a loop across its teams over
  the tile's pixels, and between tiles answers any thief. `MANDEL_POLICY`
  chooses how the tiles are shared:
  - `static`: each place gets an equal, contiguous range of the frame's
    tiles by rows, and nothing moves.
  - `steal` (the default): the static ranges; a place that runs out asks
    its neighbour round the ring, which gives it the back half of what it
    has left, or passes the request on if it has fewer than two tiles left.
  - `pool`: the host holds the frame's tiles, and a place that runs out asks
    it for `MANDEL_CHUNK` (4).

  Besides the iterations and checksum, it prints the tiles handed over,
  each place's tiles and busy time by the device's clock, and the balance:
  the busiest place's time over the average, 1 when the places are even.
- `mandel.py`, independent, in Python, whose floats are doubles that it does
  not fuse either: the counts for small frames, which the tests check the
  programs against.

`MANDEL_CENTRE=x,y` moves the centre of `mandel.c` and `mandel.py`, which
check the work-stealing example's zoom (`examples/work-stealing`) that way.

`compare-places.sh` runs `mandel.m`, `mandel.c` and `mandel-places.m` under
each policy in turns, checks that they give the same iterations, and prints
each one's median time, rate, balance and tiles handed over.

`compare-c.sh` runs `mandel.m` and `mandel.c` in turns, checks that they give
the same iterations, and prints each one's median time and rate.

2026-11-12, gfx1030, on compute units 0-39, 40 frames, the median of 5
rounds of the whole zoom's time:

| Frames | Mercury, GPU process | C, a kernel a frame |
| --- | ---: | ---: |
| 2048² | 3.45 s, 38.8 Giterations/s | 3.57 s, 37.4 Giterations/s |
| 1024² | 0.90 s, 37.0 Giterations/s | 0.98 s, 34.0 Giterations/s |

`compare-places.sh`, the same day, the median of 5 rounds; four places,
then the balance, the busiest place's busy time over the average, and the
tiles handed over:

| Frames, tiles | Mercury, dense | C, dense | Static | Steal | Pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048², 128² | 3.45 s | 3.57 s | 4.28 s, 1.23 | 3.97 s, 1.04, 2362 | 3.94 s, 1.02, 10240 |
| 2048², 64² | 3.45 s | 3.57 s | 4.77 s, 1.21 | 4.43 s, 1.02, 10005 | 4.49 s, 1.02, 40960 |
| 1024², 128² | 0.90 s | 0.98 s | 1.11 s, 1.23 | 1.09 s, 1.11, 404 | 1.06 s, 1.02, 2560 |

Stealing evens the places and saves 7% over the static split, but each
tile's loop waits for its slowest pixel, so splitting the GPU costs 9%
before any balance, and on one GPU the hardware's balancing of the dense
programs wins (design document, Measurements, Uneven work).

### On gfx90a

2026-11-14, one die of an AMD Instinct MI250 (gfx90a, all 104 compute
units), GPU 1, the host's threads on the other socket
(`numactl --cpunodebind=8-15`), the build of 693f7c3053d4,
`compare-places.sh`, 40 frames, the median of 5 rounds; four places, then
the balance and the tiles handed over:

| Frames, tiles | Mercury, dense | C, dense | Static | Steal | Pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048², 128² | 0.497 s, 269 Gi/s | 0.371 s, 361 Gi/s | 10.4 s, 1.04 | 11.9 s, 1.04, 14 | 12.8 s, 1.03, 10240 |
| 2048², 64² | 0.495 s, 271 Gi/s | 0.371 s, 361 Gi/s | 40.7 s, 1.05 | 45.8 s, 1.09, 661 | 48.9 s, 1.02, 40960 |
| 1024², 128² | 0.215 s, 155 Gi/s | 0.115 s, 290 Gi/s | 3.05 s, 1.04 | 3.98 s, 1.04, 0 | 3.63 s, 1.02, 2560 |

- **The dense programs are 4 to 7 times as fast as on gfx1030,** on 104 units against
  40, but Mercury's GPU process now trails C, by 1.34 times at 2048² and
  1.87 at 1024², where on gfx1030 it led.
- **The places were 14 to 100 times slower than the dense programs,**
  where on gfx1030 they were 1.2 to 1.4. The cause was the barrier that a
  GPU process crosses with its worker teams around each loop, at system
  scope then, which on gfx90a writes back and invalidates the L2 at every
  crossing (design document, Measurements, "Barriers between teams").

With that barrier at device scope, 2026-11-14, GPU 6, the same settings:

| Frames, tiles | Mercury, dense | C, dense | Static | Steal | Pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048², 128² | 0.409 s, 327 Gi/s | 0.374 s, 358 Gi/s | 0.718 s, 1.15 | 0.680 s, 1.04, 744 | 0.759 s, 1.03, 10240 |
| 2048², 64² | 0.409 s, 327 Gi/s | 0.373 s, 358 Gi/s | 2.48 s, 1.15 | 2.25 s, 1.02, 4067 | 2.61 s, 1.03, 40960 |
| 1024², 128² | 0.132 s, 253 Gi/s | 0.117 s, 287 Gi/s | 0.210 s, 1.15 | 0.222 s, 1.10, 77 | 0.233 s, 1.04, 2560 |

- **The places are now 1.6 to 1.9 times the dense GPU process at tiles of
  128², and 5.5 to 6.4 at 64²,** against 1.15 to 1.4 on gfx1030: what is
  left is a tile's fixed cost, about 0.2 ms on its place at 64² (40
  frames of 1024 tiles over four places), which the GPU's faster iterations leave in
  view. No arrays travel: a tile is one loop across the place's worker
  teams, whose sum the place keeps until it reports at the end, and the
  places' messages are small terms, orders, tiles and steals. Where the
  0.2 ms goes is not yet measured; the loop's start and end, each a
  crossing of the barrier with the place's worker teams, and its
  reduction are likely parts.
- **The dense GPU process gained 1.2 times too,** and is 1.09 to 1.13
  times C. These are timed by the host.

## Tests

`test/Benchmarks/mandel.test` runs `mandel.m` at the host, sequentially at
-O0 and -O2 and on the host's threads, and `mandel.c` on the host's threads,
against `mandel.py` on 64 pixels for 12 frames and 51 for 30;
`mandel-gpu.test` both on the GPU, the whole zoom at 512 pixels too.
`mandel-places.test` runs `mandel-places.m`'s places at the host under every
policy, with one, three and four places, sequentially at -O0 and -O2 and on
the host's threads, against `mandel.py`'s counts, and the static split's
tiles; `mandel-places-gpu.test` every policy with one to four places on the
GPU, and the whole zoom at 512 pixels.
