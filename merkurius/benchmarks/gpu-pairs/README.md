# Arrays between GPUs

`gpupairs.m` measures arrays of `int` sent between GPU processes on two GPUs
of one node, and between two GPU processes on one GPU. It is the baseline for
the direct path between GPUs (docs/INPROGRESS.md, Direct sends between the
GPUs of a node). Today every message from one GPU to another goes through the
host.

## What it measures

A relay is a GPU process with no worker teams. It forwards each array it
receives to the mailbox it was given. `main` sends arrays into a chain of
relays and receives them at the far end:

| Chain | Route |
| --- | --- |
| `A` | host → relay on GPU 0 → host |
| `A-B` | host → relay on GPU 0 → relay on GPU 1 → host |
| `A-A` | host → relay on GPU 0 → relay on GPU 0 → host |

`A-B` less `A` is about one hop from one GPU to another. `A-A` less `A` is
one hop within a GPU, through the queue between processes of one kernel. In
`ping` one array is in flight, which measures latency. In `stream` all of a
measurement's arrays are sent before any is received, which measures
bandwidth.

Each measurement moves 64 MiB (`GPU_PAIRS_MIB`) in 2 to 20000 arrays, and at
most 2000 for `ping`. Sizes run from 1 to 2^20 words. `right` counts the
arrays that arrived with the right size and last element. The time is taken
on the host.

## Running it

```
utils/gpu/with-gpus.py --gpus A,B -- benchmarks/gpu-pairs/run.sh [build-dir] [offload-arch]
```

GPU 0 and GPU 1 of the program are A and B. If only one GPU is visible,
`A-B` is left out.

## On gfx90a

Measured on r11 (8 MI250 GCDs, all linked by xGMI) on 2026-11-14 at
8a59002cabdd. The host threads ran on the other socket
(`numactl --cpunodebind=8-15`). The pair is ROCR GPUs 1 and 3, on different
packages, one xGMI hop apart. All 54 rows had every array right on all four
pairs measured (2,3 in one package; 1,3; 1,2; 1,7).

| Chain | Mode | 1 word | 4096 words | 65536 words | 2^20 words |
| --- | --- | --- | --- | --- | --- |
| `A` | ping | 96 µs | 129 µs, 253 MB/s | 604 µs, 868 MB/s | 7.7 ms, 1.09 GB/s |
| `A-B` | ping | 179 µs | 226 µs, 145 MB/s | 699 µs, 750 MB/s | 12.5 ms, 669 MB/s |
| `A-A` | ping | 160 µs | 391 µs, 84 MB/s | 4.2 ms, 126 MB/s | 98 ms, 86 MB/s |
| `A` | stream | | 183 MB/s | 1.01 GB/s | 1.36 GB/s |
| `A-B` | stream | | 191 MB/s | 1.04 GB/s | 846 MB/s |
| `A-A` | stream | | 67 MB/s | 94 MB/s | 87 MB/s |

What it shows:

- One hop between GPUs costs about 80 µs (`A-B` less `A` at 1 word). For
  large arrays it costs as much as a round trip through the host does,
  because it is one: GPU A copies to host memory, and the host copies to
  GPU B.
- One hop within a GPU costs about 65 µs at 1 word. Large arrays then move
  at about 90 MB/s, because the sending relay has no worker teams and one
  lane copies the array for the queue.
- Pair 2,3 (one package) and pairs 1,2 and 1,7 give the same numbers to
  within about 20%. None of the hardware's links are used.
- For comparison, HIP probes on the same pairs measured 38 to 150 GB/s
  between the GPUs, and about 1.2 µs one way for a flag between running
  kernels.

**With the wave's lanes copying** (docs/INPROGRESS.md, Direct sends between
the GPUs of a node, step 0), measured the same day on GPUs 1,3. `A` is
unchanged, within 5%. `A-B`'s stream of 2^20 words gave 658 MB/s against
846 before, within the spread across pairs before the change (846 to 1480
MB/s); `A-B` does not copy within a GPU.

| Chain | Mode | 1 word | 4096 words | 65536 words | 2^20 words |
| --- | --- | --- | --- | --- | --- |
| `A-A` | ping | 164 µs | 174 µs, 189 MB/s | 847 µs, 619 MB/s | 17.0 ms, 494 MB/s |
| `A-A` | stream | | 186 MB/s | 764 MB/s | 652 MB/s |
