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
tree at -O2. MB/s, the median of three runs. Every message arrived, of the
right size, in every run.

`to`, from node 1:

| Words | `memcpy` | `gpu-local` | `cpu-local` | `gpu` | `cpu` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8 | 0.4 | 4.8 | 0.3 | 1.6 |
| 16 | 12.5 | 5.5 | 76 | 5.2 | 25 |
| 256 | 201 | 86 | 840 | 72 | 289 |
| 4096 | 2294 | 614 | 1223 | 641 | 702 |
| 65536 | 7836 | 847 | 1309 | 786 | 715 |
| 262144 | 8348 | 1067 | 1314 | 335 | 728 |
| 1048576 | 7347 | 1024 | 1301 | 380 | 590 |

`from`, to node 1:

| Words | `memcpy` | `gpu-local` | `gpu` | `cpu` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9 | 0.3 | 0.3 | 1.5 |
| 16 | 14 | 5.0 | 4.0 | 23 |
| 256 | 226 | 79 | 66 | 264 |
| 4096 | 3392 | 1133 | 472 | 638 |
| 65536 | 10583 | 4243 | 822 | 790 |
| 262144 | 20123 | 7155 | 742 | 664 |
| 1048576 | 23689 | 11438 | 640 | 517 |

What they show:

- **Small messages are latency.** Up to a few hundred words, a message costs a
  fixed time: from the GPU within a node, about 20 µs to send and 25 µs to
  receive, where `omp_target_memcpy` alone takes about 10 µs.
- **Sending within a node stops at about 1 GB/s, and the GPU is not why.** The
  copy from the device alone reaches 7 to 8 GB/s, but `gpu-local` stays near
  1 GB/s, and `cpu-local`, with no GPU, near 1.3 GB/s. What they share is the
  send's own copy of the message into its fragment, by the array's
  `type_info`, an element at a time, into newly allocated memory. It was not
  measured apart.
- **Receiving within a node has no such copy,** since the host copies the
  message straight into the device's arena, and reaches 11 GB/s, half the
  floor.
- **Between nodes, the GPU and the CPU are about equal up to 65536 words,** at
  500 to 800 MB/s, so the limit there is on the host: the encoding and TCP.
- **Sending from the GPU to node 2 falls at 2^18 words and above,** to 335 to
  380 MB/s, where the CPU keeps 590 to 730. The callback writes to TCP on the
  plugin's server thread, which flow control can stall, which may be why, but
  this was not measured apart.
