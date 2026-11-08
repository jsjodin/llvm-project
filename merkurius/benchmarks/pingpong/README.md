# Ping-pong latency

`pingpong.m` measures the latency of a message round trip between
Merkurius processes, from a GPU's target region and from the CPU: the
first benchmark of the device's side of the first slice
(docs/INPROGRESS.md, step 5). Each round, node 1 sends N to its partner
and waits for N + 1. The rounds are timed on the host with
`clock_gettime`, after 100 rounds to warm up, and the time of a round is
their average.

## What it measures

| Name | Node 1's side | Its partner |
| --- | --- | --- |
| `cpu` | `main` | node 2's `main`, another process, over TCP on loopback |
| `gpu` | a target region on the GPU | node 2's `main`, over TCP on loopback |
| `cpu-local` | `main` | an echo thread in node 1, in C, through the runtime's mailboxes |
| `gpu-local` | a target region on the GPU | the echo thread |

A target region's time includes its launch, which `launch`, an empty
region's time, shows; over 10000 rounds it is a few nanoseconds a round.

Beside them are the floors, plain C round trips with nothing of ours:

| Name | What |
| --- | --- |
| `tcp-floor.c` | two processes over a TCP connection on loopback, with `TCP_NODELAY`, as node 1 and node 2 are |
| `condvar-floor.c` | two threads through a mutex and condition variables, as a mailbox waits |
| `rpc-floor.cpp` | a target region's round trip over the offload plugin's RPC server, LLVM libc's protocol, with a callback that adds one: what each of the device's sends and polls costs |

## Building and running

`run.sh [build-dir] [offload-arch]` builds everything and runs each
measurement once. The build directory defaults to `build-merk`, beside
`llvm-project`; without an offload-arch the GPU's measurements are left
out. `PINGPONG_ROUNDS` sets the rounds, 10000 by default.

```sh
merkurius/benchmarks/pingpong/run.sh "" gfx1030
```

`pair.py PROGRAM cpu|gpu ROUNDS` runs node 2 and node 1 against each other,
with a cookie of their own and a temporary HOME. Nodes 1 and 2 are the same
program, `pingpong partner` and `pingpong cpu|gpu PORT ROUNDS`; the local
ones are `pingpong cpu-local|gpu-local ROUNDS`.

## Results

2026-11-07, an AMD Radeon RX 6950 XT (gfx1030) on PCIe, an AMD Ryzen
Threadripper PRO 5965WX (24 cores, 48 threads) as the host, the build of this tree at -O2, with libc on the GPU. Microseconds a
round, 20000 rounds, three runs:

| Name | Runs | |
| --- | --- | --- |
| `cpu` | 44.0, 44.6, 34.6 | node 2 over TCP |
| `gpu` | 56.5, 47.0, 43.3 | node 2 over TCP, from the GPU |
| `cpu-local` | 14.4, 16.2, 16.1 | the echo thread |
| `gpu-local` | 27.1, 27.9, 34.1 | the echo thread, from the GPU |
| `tcp-floor` | 15.4, 15.1, 15.3 | |
| `condvar-floor` | 8.8, 9.1, 9.0 | |
| `rpc-floor` | 7.5, 7.6, 7.5 | |

The runs vary by up to a third from one to the next, more with TCP, so
only differences of several microseconds mean anything. What they show:

- **The GPU's hop costs about as much as one more trip between threads.**
  From the GPU, a round with node 2 takes 5 to 12 µs more than from the
  CPU, and a round with the echo thread about 12 µs more.
- **That hop is the RPC round trips.** A send waits for the host to queue
  the message, one round trip of 7.5 µs, and a receive polls the host,
  one round trip each, until the answer is there: the answer comes in the
  middle of a poll, on average, and its poll ends the round. Shortening the
  device's sleep between polls from about 8000 cycles to 64 changed
  nothing beyond the noise (`s_sleep` 127, 16 and 1 were each tried three times), since the
  poll itself is what waits. A queue in device memory (step 7), which the
  host writes and the device reads without a round trip, removes the
  polls; a send that does not wait for the host, ordered by a single port,
  would remove the other.
- **The plugin's RPC server sleeps on an interrupt.** When a scan finds
  no work, the AMD plugin's server thread blocks on an HSA signal, which
  the GPU's next request rings, so each round trip after a pause pays for
  the interrupt and the thread's wake-up; that is in the floor. (This is
  from reading the plugin, `RPC.cpp` and the AMD plugin's `rtl.cpp`; it was
  not measured apart.)
- **Between nodes, our runtime costs twice TCP's.** A round over TCP takes
  about 45 µs from the CPU, where a plain round trip over loopback takes
  15. On each node a message crosses from the connection's reader thread
  to the receiver, a trip between threads of about 9 µs (`condvar-floor`),
  and is encoded and decoded. A receiver that reads the connection itself,
  or that spins briefly before it blocks, would save the trips.
- **Mailboxes cost about 3 µs a message.** A round with the echo thread
  takes 14 to 16 µs from the CPU, where two threads take 9: the rest is
  copying the message, queueing it and the receive's checks.

### With the spin before blocking

2026-11-08, the same machine, with a mailbox's ring for the device and the
host's threads spinning before they block. Microseconds a round, 20000
rounds, five runs, the ranges with `MERK_SPIN_MICROSECONDS=0` and with the
default of 50:

| Name | No spin | Spin |
| --- | --- | --- |
| `gpu` | 47.5 to 67.7 | 33.9 to 36.0 |
| `cpu` | 38.5 to 64.4 | 21.9 to 25.7 |
| `gpu-local` | 23.3 to 25.8 | 19.2 to 19.4 |
| `cpu-local` | 11.8 to 16.7 | 6.8 to 6.9 |

The spin yields, so `cpu` with both nodes pinned to one core takes about 28
µs with it, and about 39 without.
