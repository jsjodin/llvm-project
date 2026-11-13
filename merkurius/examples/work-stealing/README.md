# Work stealing

Four workers share a GPU and a job whose pieces cost very different amounts.
Each worker starts with an equal share of the pieces. A worker that runs out
steals from a neighbour, so that none sits idle while others still work.

The job is a zoom into the Mandelbrot set: 40 frames, each cut into square
tiles. A pixel costs as many iterations as it takes to escape, and pixels in
the set never escape. So some tiles are cheap and some are costly, and which
ones changes from frame to frame. The zoom's centre was chosen from some 300
points near the set's boundary as the one whose frames an equal split leaves
most uneven.

- The workers are *places*: GPU processes on one GPU, each with a quarter of
  the GPU's teams. A place computes one tile at a time, as a loop that runs
  in parallel across its teams.
- For each frame the host gives every place an equal range of tiles, and
  waits for all of them to report.
- A place that runs out of tiles sends a `steal` message to its neighbour in
  a ring of places. A neighbour with two or more tiles left gives the thief
  the back half of them. One with fewer passes the request on round the
  ring. When it comes back round to the thief, the thief is done for the
  frame.
- A place checks for thieves between two tiles, without waiting.

Everything is in `work_stealing.m`. Its first part is the host, its second
the places, each state of a place a predicate that calls the next as its last
step, and its third the Mandelbrot set.

## Running it

From `examples/`, with `merkc` built:

    make run-work-stealing
    make run-work-stealing-no-steal

The first runs it with stealing, the second without, for comparison. Both
build `build/work_stealing` at `-O2` for the GPU that `ARCH=` names,
`gfx1030` by default. `WHERE=host` runs the places on the host's CPUs
instead. The program itself takes the same choices as arguments:

    work_stealing [no-steal] [host]

Environment variables change the job:

| Variable | Default | |
| --- | ---: | --- |
| `MANDEL_SIZE` | 2048 | a frame's side, in pixels |
| `MANDEL_TILE` | 64 | a tile's side, in pixels; it must divide the frame's |
| `MANDEL_FRAMES` | 40 | the frames |
| `MANDEL_PLACES` | 4 | the places |

## What it prints

    Work stealing: 4 places, stealing on
    Frames: 40 of 2048 x 2048 pixels, tiles of 64 x 64
    Iterations: 121437819024
    Checksum: 3709558869501
    Steals: 436
    Place 0: 13136 tiles, 194 steals
    Place 1: 12108 tiles, 151 steals
    Place 2: 7450 tiles, 36 steals
    Place 3: 8266 tiles, 55 steals
    Time: 2.048 s

- *Iterations* and *Checksum* are the same with and without stealing, and
  for any number of places: every tile is computed once.
- *Steals* counts the steals that got tiles; each place's line shows its
  tiles and its steals over the whole zoom. Without stealing each place
  computes its own share, here 10240 tiles.
- *Time* is the host's, from starting the places to their last report.

## How much stealing saves

On an RX 6950 XT (gfx1030), 2026-11-12, four places, one run each way:

| Frames | Tiles | Stealing | Without | Saved |
| --- | --- | ---: | ---: | ---: |
| 1024² | 64² | 0.57 s | 0.60 s | 5% |
| 2048² | 64² | 2.05 s | 2.27 s | 10% |
| 2048² | 128² | 1.99 s | 2.17 s | 8% |
| 2048² | 256² | 1.71 s | 1.79 s | 5% |
| 4096² | 64² | 7.71 s | 8.65 s | 11% |

- *Larger frames* give stealing more to even out. A tile of the same size
  covers less of the frame, so the costly pixels fall into fewer tiles,
  and the places' shares differ more.
- *Larger tiles* are faster, with or without stealing: each tile starts and
  ends a loop across the place's teams, which costs the same whatever the
  tile's size. But with fewer tiles there is less to share out.
- *The largest sensible frame* is 8192², some 30 s. In the last frame,
  neighbouring pixels are then 3 units in the last place of a double
  apart; at 16384² they would be 1.5, and start to fall on the same point.

The example shows that stealing works, not that it is the fastest way to use
one GPU: a single loop over a whole frame, which the GPU's hardware balances
by itself, is faster (`benchmarks/mandel/README.md`). Stealing is for work
spread over several GPUs, where no hardware balances it.

## Tests

`test/Examples/work-stealing.test` runs it at the host, with and without
stealing, against independently computed totals; `work-stealing-gpu.test` on
the GPU.
