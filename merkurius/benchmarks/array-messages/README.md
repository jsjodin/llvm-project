# Arrays as messages

`arraymessages.m` measures the bandwidth of arrays as messages between
Merkurius processes, from a GPU's target region and from the CPU, over
message sizes of 1 to 2^20 words (docs/INPROGRESS.md, The device's side of
the first slice, step 6). The bandwidth of memory itself is the other
benchmark, [`bandwidth`](../bandwidth/README.md).

## What it measures

Each measurement moves 64 MiB, in at least 2 messages and at most 20000, in
one direction: `to`, node 1 sends the arrays; `from`, node 1 receives them.

| Name | Node 1's side | Its partner |
| --- | --- | --- |
| `cpu` | `main` | node 2's `main`, another process, over TCP on loopback |
| `gpu` | a target region on the GPU | node 2's `main`, over TCP on loopback |
| `cpu-local` | `main` | its own mailbox |
| `gpu-local` | a target region on the GPU | `main`, before or after the region |
| `memcpy` | `omp_target_memcpy` alone, from `main`, no kernel running (`memcpy-floor.cpp`) | |

The time is taken on the host, less that of a run with no messages, which has
the region's launch and the making of the array. Over TCP, `to` includes node
2's answer once it has them all. Within a node, `main` sends the arrays before
the time, for `from`, or receives them after it, for `to`, so a region's time
is the region's alone. `cpu-local from` only takes messages already queued,
with no copy, so it measures nothing but the queue. For `memcpy`, `to` is from
the device to the host, as a send from the GPU is.

## Building and running

`run.sh [build-dir] [offload-arch]` builds everything and runs each
measurement once. `ARRAY_MESSAGES_MIB` sets the mebibytes of each measurement.
`../pingpong/pair.py PROGRAM cpu|gpu MIB` runs node 2 and node 1 against each
other.

```sh
merkurius/benchmarks/array-messages/run.sh "" gfx1030
```

## Results

2026-11-08, an AMD Radeon RX 6950 XT (gfx1030) on PCIe, an AMD Ryzen
Threadripper PRO 5965WX (24 cores, 48 threads) as the host, the build of this
tree at -O2, with arrays of scalars copied and encoded as blocks. MB/s, the median of three runs. Every message arrived, of the
right size, in every run.

`to`, from node 1:

| Words | `memcpy` | `gpu-local` | `cpu-local` | `gpu` | `cpu` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8 | 0.4 | 4.9 | 0.3 | 1.4 |
| 16 | 13 | 5.9 | 79 | 5.0 | 23 |
| 256 | 203 | 93 | 1240 | 68 | 319 |
| 4096 | 2550 | 753 | 2151 | 917 | 1242 |
| 65536 | 8565 | 1471 | 2008 | 1256 | 1422 |
| 262144 | 8360 | 1174 | 2199 | 1504 | 1662 |
| 1048576 | 7338 | 1184 | 2234 | 1127 | 1416 |

`from`, to node 1:

| Words | `memcpy` | `gpu-local` | `gpu` | `cpu` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9 | 0.3 | 0.3 | 1.4 |
| 16 | 14 | 5.0 | 4.2 | 23 |
| 256 | 226 | 79 | 68 | 310 |
| 4096 | 3397 | 1098 | 677 | 1271 |
| 65536 | 10539 | 4181 | 1179 | 1252 |
| 262144 | 20404 | 6572 | 1242 | 1363 |
| 1048576 | 23715 | 10384 | 821 | 680 |

Runs between nodes vary by up to a third from one to the next.

What they show:

- **Small messages are latency.** Up to a few hundred words, a message costs a
  fixed time: from the GPU within a node, about 20 µs to send and 25 µs to
  receive, where `omp_target_memcpy` alone takes about 10 µs.
- **Sending within a node stops at about 1.2 GB/s from the GPU and 2.2 GB/s
  from the CPU,** where the copy from the device alone reaches 7 to 8 GB/s.
  The GPU's send copies the array twice on the host, from the device into a
  temporary and into the message's fragment, both newly allocated memory; the
  CPU's copies it once. Copying arrays of scalars as a block raised these from
  1.0 and 1.3 GB/s, and it was the array's copy an element at a time by its
  `type_info` that had limited them.
- **Receiving within a node has no second copy,** since the host copies the
  message straight into the device's arena, and reaches about 10 GB/s, near
  half the floor.
- **Between nodes, from 65536 words up, the GPU and the CPU are about equal,**
  at about 0.7 to 1.7 GB/s, so the limit is on the host: the encoding and
  TCP. At 4096 words a receive on the GPU, at 677 MB/s, is half the CPU's. Sending 2^18
  words or more from the GPU fell to 335 to 380 MB/s while arrays were encoded
  an element at a time; encoded as a block they reach 1.1 to 1.5 GB/s, so the
  callback's TCP writes were not what limited them.
