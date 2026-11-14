# A barrier between teams

`teambarrier.m` measures a barrier between the teams of a target teams region,
made from team mailboxes (docs/INPROGRESS.md, Processes as teams, step 5), and
the library's `process.team_barrier` (design document, Library code on the
device). Every team arrives, and is released once all have. The shapes from
messages are ordinary Mercury code of the program, which calls the library's
`list.length` on the device.

## What it measures

| Shape | A round |
| --- | --- |
| `central` | Each team sends its arrival to team 0, which then sends each a release. Team 0 receives and sends one message per team, so a round takes about twice as many hops in sequence as there are teams. |
| `tree` | A binary tree of the teams: a team waits for its children's arrivals, sends its own to its parent, waits for its parent's release and passes it on. A round takes about twice the tree's depth in hops in sequence. |
| `library` | `process.team_barrier`: a counter in the region's team table that each team adds to, and the round that the last to arrive ends, which the others wait for. No messages. |

In the shapes from messages, an arrival in round R is the message 2R, a release
2R + 1. Each team counts the messages it did not expect, or that did not come
within 10 s, and reports them to the host, which prints their sum as `wrong`.
The `library` shape reports 0.

The rounds are timed on the host with `clock_gettime`, after 100 rounds to warm
up, less the time of a region of no rounds. That holds the launch and the teams'
wait for each other to start. A barrier's time is the rest divided by the
rounds.

## Building and running

`run.sh [build-dir] [offload-arch]` builds the program and runs the three shapes,
over numbers of teams, on the GPU and on the host's teams, which libomp runs as
threads. The build directory defaults to `build-merk`, beside `llvm-project`.
Without an offload-arch, the GPU's measurements are left out.
`TEAM_BARRIER_ROUNDS` sets the rounds, 10000 by default, and
`TEAM_BARRIER_TEAMS` the numbers of teams, `2 4 16 64` by default.

```sh
merkurius/benchmarks/team-barrier/run.sh "" gfx1030
```

`teambarrier central|tree|library TEAMS ROUNDS` runs one measurement.

## Results

2026-11-10, an AMD Radeon RX 6950 XT (gfx1030) on PCIe, an AMD Ryzen
Threadripper PRO 5965WX (24 cores, 48 threads) as the host, the build of
801205c01704 with this benchmark at -O2. Microseconds a barrier, 10000 rounds,
three runs, none with a wrong message:

| Shape | Teams | GPU | Host's teams |
| --- | ---: | --- | --- |
| `central` | 2 | 3.83, 3.79, 3.77 | 2.03, 2.03, 2.00 |
| `central` | 4 | 6.82, 6.71, 6.72 | 8.57, 8.15, 7.22 |
| `central` | 16 | 33.88, 33.44, 33.27 | 90.12, 102.71, 86.33 |
| `central` | 64 | 154.21, 152.90, 150.53 | 2251.00, 2233.09, 2271.86 |
| `tree` | 2 | 3.83, 3.74, 3.73 | 1.15, 2.11, 2.05 |
| `tree` | 4 | 8.11, 7.99, 7.91 | 9.22, 8.53, 8.61 |
| `tree` | 16 | 17.40, 17.00, 17.10 | 49.73, 53.69, 55.80 |
| `tree` | 64 | 28.83, 28.72, 28.71 | 1571.96, 1576.28, 1535.22 |

On the GPU, a hop of a message in a round costs about 1.2 µs. That is 5 to 10
times the 240 ns of a word written to memory and polled, which the estimates
in docs/INPROGRESS.md assumed. The tree is 5 times faster than central at 64
teams. The host's teams are threads waking each other through condition
variables, and at 64 there are more of them than hardware threads. The design
document's Measurements section has what is learned from it.

2026-11-10, the same machine, the build of 6a90cadd2fac, with the `library`
shape (`process.team_barrier`), three runs, none with a wrong message:

| Shape | Teams | GPU | Host's teams |
| --- | ---: | --- | --- |
| `central` | 2 | 3.79, 3.76, 3.76 | 2.02, 1.12, 2.02 |
| `central` | 4 | 6.85, 6.79, 6.87 | 8.92, 9.76, 9.11 |
| `central` | 16 | 34.00, 34.33, 33.61 | 124.32, 64.96, 105.69 |
| `central` | 64 | 150.30, 151.42, 152.19 | 2264.41, 2246.40, 2264.84 |
| `tree` | 2 | 3.73, 3.79, 3.74 | 2.04, 2.01, 2.02 |
| `tree` | 4 | 8.04, 7.93, 7.89 | 8.40, 9.18, 7.58 |
| `tree` | 16 | 17.24, 17.08, 17.15 | 48.16, 56.73, 46.91 |
| `tree` | 64 | 28.86, 28.78, 28.80 | 1522.38, 1541.47, 1580.33 |
| `library` | 2 | 0.68, 0.74, 0.68 | 0.31, 0.24, 0.27 |
| `library` | 4 | 0.76, 0.75, 0.78 | 1.09, 0.70, 1.31 |
| `library` | 16 | 0.80, 0.78, 0.79 | 10.48, 8.10, 10.08 |
| `library` | 64 | 1.00, 0.96, 0.97 | 36.42, 35.43, 35.14 |

The library's barrier, a counter in the runtime, takes 1 µs or less on the GPU
at every number of teams, 30 times faster than the tree at 64.

### On gfx90a

2026-11-14, one die of an AMD Instinct MI250 (gfx90a, 104 compute units),
two AMD EPYC 7713 (128 cores), the build of 693f7c3053d4, three runs of
`run.sh` with the default rounds, µs a barrier:

| Shape | Teams | GPU | Host's teams |
| --- | ---: | --- | --- |
| `central` | 2 | 9.61, 9.38, 9.55 | 1.39, 1.25, 1.37 |
| `central` | 4 | 17.12, 16.81, 17.22 | 5.11, 7.94, 8.43 |
| `central` | 16 | 177.12, 180.13, 179.95 | 166.68, 162.23, 164.48 |
| `central` | 64 | 2382.50, 2843.87, 3046.95 | 4080.46, 4129.22, 4111.21 |
| `tree` | 2 | 9.40, 9.60, 9.22 | 1.37, 1.38, 1.41 |
| `tree` | 4 | 19.80, 19.50, 19.25 | 4.85, 7.86, 8.30 |
| `tree` | 16 | 88.13, 87.91, 87.49 | 69.38, 114.92, 91.96 |
| `tree` | 64 | 520.47, 515.55, 518.86 | 1389.26, 1362.26, 1347.69 |
| `library` | 2 | 2.14, 2.38, 2.43 | 0.27, 0.25, 0.26 |
| `library` | 4 | 2.62, 2.74, 2.62 | 1.06, 1.06, 1.03 |
| `library` | 16 | 7.62, 7.55, 7.73 | 13.42, 10.20, 8.99 |
| `library` | 64 | 30.36, 30.91, 31.03 | 57.46, 59.68, 57.52 |

The host's teams are about as on the other machine. On the GPU every shape
is slower than on gfx1030, by 2.5 to 3 times at 2 teams and by 18 to 30
times at 64: `library` 30 µs against 1.0, `tree` 520 against 29. The cause
is the device runtime's atomics, which are `__atomic_*` and so at system
scope. On gfx90a, whose L2 is not coherent with the host, a release at
system scope writes back the whole L2 (`buffer_wbl2`) and an acquire
invalidates it (`buffer_invl2`), each poll of a wait included; on gfx1030
the code is the same at either scope. With `library`'s counters at device
scope (`__scoped_atomic_*` with `__MEMORY_SCOPE_DEVICE`, which is right, as
only the teams touch them), a trial build ran it in 1.33 µs at 2 teams and
1.43 at 64. The other shapes go through team mailboxes, whose rings and
allocations wait on the same kind of atomics. The host's NUMA node did not
matter: driving GPU 0 from the cores of its own node, 4, or of node 12, on
the other socket, gave the same times.

With the barrier's counters at device scope (`ArriveAndWait` in
`runtime/device/device.cpp`, 2026-11-14), one run: `library` 1.21, 1.28,
1.38 and 1.43 µs at 2, 4, 16 and 64 teams, 1.2 to 1.5 times gfx1030's;
`central` and `tree` as before, as their mailboxes are not changed yet.
These benchmarks are timed by the host, so the device clock's error
(design document, Measurements, "Barriers between teams") did not touch
them.
