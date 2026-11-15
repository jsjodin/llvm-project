# Arrays over the link between two GPUs

`link.m` measures arrays of `int` streamed from GPU processes on one GPU to
GPU processes on another, with no host on the way. The host only starts
each measurement and collects the results. gpu-pairs (../gpu-pairs) starts
and ends every chain at the host, so its large arrays are limited by the
host's legs. This benchmark measures the direct path between GPUs
(docs/INPROGRESS.md, Direct sends between the GPUs of a node) against the
link itself.

## What it measures

A source makes an array of WORDS words once and sends it COUNT times to its
sink. The sink checks each array's size and last element and times the
stream on its own GPU's clock, from the first array to the last.

| Case | Pairs |
| --- | --- |
| `A>A` | 1 source and its sink, both on GPU 0 |
| `A>B` | 1 source on GPU 0, its sink on GPU 1 |
| `A>Bx4` | 4 such pairs at once |
| `A<>B` | 1 pair each way at once |
| `A<>Bx4` | 4 pairs each way at once |

Each measurement moves 256 MiB (`GPU_LINK_MIB`), split between the pairs,
in at most 20000 arrays a pair. Sizes run from 1 to 2^20 words (8 MiB).
Before it starts the clock, the host waits until each source's first word
has reached its sink, so that the kernels are running and the routes are
made.

- `device MB/s`: for each GPU with sinks, the arrays received there over
  the span on that GPU's clock, from the earliest first array to the
  latest last, summed over the GPUs. Two GPUs' clocks are not compared.
- `host MB/s`: every array over the host's time, from telling the sources
  to start to the last sink's report. It agrees with `device MB/s` except
  where a measurement has only a few arrays.
- `us each`: the mean time between arrays at a sink.

## Running it

```
utils/gpu/with-gpus.py --gpus A,B -- benchmarks/gpu-link/run.sh [build-dir] [offload-arch]
```

With one GPU visible, only `A>A` runs. Without `MERK_GPU_PEER=1`, the
arrays go through the host. `GPU_LINK_WORKERS=K` gives each source and sink
worker teams, up to a K'th of its GPU's kernel's teams (`wsource`,
`wsink`). It builds with `-fopenmp`, which they need, and from 64 KiB the
workers copy the arrays instead of the copy engine (docs/INPROGRESS.md,
step 7).

test/Benchmarks/gpu-link-two-gpus.test runs it on two GPUs at 1 MiB a
measurement, both without workers and with 8. It checks that every array
arrived right and that the messages went directly.

## Results

Measured on r11 on 2026-11-15 on ROCR GPUs 4,6, which are in different
MI250 packages, with the host threads on the other socket. The columns are:

- "host": without `MERK_GPU_PEER`.
- "direct": `MERK_GPU_PEER=1`, so from 64 KiB the copy engine copies the
  arrays.
- "engine": 8 worker teams per process, but `MERK_GPU_PEER_WORKERS=0`.
- "workers": 8 worker teams per process, which copy first.

GB/s by `device MB/s`; every array arrived right in all four.

| Case | Words | host | direct | engine | workers |
| --- | --- | --- | --- | --- | --- |
| `A>A` | 65536 | 4.7 | 4.9 | 18.3 | 18.9 |
| `A>A` | 2^20 | 3.3 | 3.4 | 85.1 | 85.4 |
| `A>B` | 65536 | 4.3 | 7.3 | 7.3 | 9.9 |
| `A>B` | 262144 | 2.8 | 20.9 | 21.0 | 24.9 |
| `A>B` | 2^20 | 2.9 | 35.7 | 35.8 | 36.2 |
| `A>Bx4` | 65536 | 3.4 | 13.5 | 13.4 | 23.2 |
| `A>Bx4` | 2^20 | 2.4 | 33.1 | 32.9 | 36.0 |
| `A<>B` | 262144 | 9.2 | 26.5 | 26.6 | 47.1 |
| `A<>B` | 2^20 | 9.4 | 37.1 | 36.4 | 68.7 |
| `A<>Bx4` | 65536 | 4.5 | 13.9 | 13.1 | 44.8 |
| `A<>Bx4` | 2^20 | 2.1 | 28.6 | 30.8 | 55.3 |

µs between arrays at a sink, one pair (`A>B`), direct:

| Words | 1 | 1024 | 4096 | 16384 | 65536 | 262144 | 2^20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| µs | 33 | 41 | 50 | 63 | 72 | 100 | 235 |

What it shows:

- **One direction reaches the link.** At 8 MiB arrays one pair streams 36
  GB/s from GPU to GPU, by the copy engine or by workers. HIP's probes
  (../gpu-pairs/peer.hip) gave 38–42 GB/s between packages.
- **Both directions together are where the copy engine falls short.** The
  host's RPC server makes one copy-engine copy at a time, for all GPUs. So
  however many pairs run, in one direction or both, the engine's total
  stays at 28–37 GB/s. With workers, `A<>B` reaches 69 GB/s, about the
  link both ways, and four pairs in one direction reach 23 GB/s at 64 KiB
  against the engine's 13.
- **Within a GPU,** a process without workers copies with its one wave,
  3–5 GB/s. Workers reach 85 GB/s.
- **Small messages cost 33–55 µs each, wherever they go.** Even an array
  of 1 word took 33 µs between GPUs, and 52 µs within one, so a pair
  managed about 30000 messages a second. The flag latency between GPUs is
  about 1.2 µs (peer.hip), so this cost was in the processes' own send and
  receive, not in the link. After the changes in docs/INPROGRESS.md (The
  cost of a message), an array of 1 word takes 17 µs between GPUs and 17
  within one, and a word 9 and 11.
- Through the host, large arrays manage 2–9 GB/s, and small ones cost
  50–90 µs.
