# A pipeline of processes

A microbenchmark of streaming through GPU processes (`docs/INPROGRESS.md`,
A pipeline of processes): how much work a frame must cost a stage before
passing it on to the next process pays, and what feedback from a later
stage to an earlier one costs.

## What it does

`pipeline.m` has three GPU processes, each with a third of the GPU's teams:

- **the source** makes each frame of `PIPE_WORDS` floats and does the first
  stage's work on it;
- **the middle process** does the second stage's;
- **the sink** does the third's and the frame's checksum.

Each passes the frame on in a message, `frame(F, Cells)`. A stage's work on
a cell is `PIPE_WORK` rounds of `x = x * a + b`, with its own `a` and `b`,
in a loop under `distribute parallel do` across the process's teams. With
`PIPE_FEEDBACK=D` above 0, the sink sends each frame's checksum back to the
source, which waits for frame F - D's before making frame F and mixes it
into the frame's cells, so at most D frames are in flight. `PIPE_MODE=single`
runs the three stages one after another in one GPU process, which has all
the teams, with no messages.

`pipeline.hip` is the same stages as three kernels a frame in one stream,
the bar that HIP sets. Its feedback is free: the kernel that makes frame F
reads frame F - D's checksum from the GPU's memory, where the stream's order
has put it.

All give the same checksum: the sum, modulo 2^64, over the frames and their
cells of `bits(x) (2 i + 1)`, `i` the cell's index, which no order changes.
Each prints the time a frame, the average from the first frame's end to the
last's: by the sink's clock, or the single process's, or HIP's events.

The settings are `PIPE_WORDS` (65536), `PIPE_WORK` (64), `PIPE_FRAMES` (200),
`PIPE_FEEDBACK` (0) and `PIPE_MODE` (`pipeline` or `single`).

## Building and running

`sweep.sh [build-dir] [offload-arch] [rounds]` builds both at -O2, runs
every configuration of `PIPE_SIZES`, `PIPE_WORKS` and `PIPE_FEEDBACKS` in
the three ways, taking turns, checks that each configuration gives one
checksum, and prints the median time a frame:

```sh
python3 -I merkurius/utils/gpu/with-gpus.py --count 1 -- \
  merkurius/benchmarks/pipeline/sweep.sh "" gfx1030
```

`pipeline host` spawns the processes at the host, for tests. Results are in
the design document, Measurements, "A pipeline of GPU processes".

## Tests

`test/Benchmarks/pipeline.test` runs the processes at the host, both modes
with and without feedback, at -O0 and -O2; `pipeline-gpu.test` runs them on
the GPU, and `pipeline.hip`, against the same checksums.
