# A barrier between teams

`teambarrier.m` measures a barrier between the teams of a target teams region,
made from team mailboxes (docs/INPROGRESS.md, Processes as teams, step 5).
Every team arrives, and is released once all have. It is ordinary Mercury code
of the program, as a barrier in the library would be. Library code is not yet
compiled for the device.

## What it measures

| Shape | A round |
| --- | --- |
| `central` | Each team sends its arrival to team 0, which then sends each a release. Team 0 receives and sends one message per team, so a round takes about twice as many hops in sequence as there are teams. |
| `tree` | A binary tree of the teams: a team waits for its children's arrivals, sends its own to its parent, waits for its parent's release and passes it on. A round takes about twice the tree's depth in hops in sequence. |

An arrival in round R is the message 2R, a release 2R + 1. Each team counts the
messages it did not expect, or that did not come within 10 s, and reports them
to the host, which prints their sum as `wrong`.

The rounds are timed on the host with `clock_gettime`, after 100 rounds to warm
up, less the time of a region of no rounds. That holds the launch and the teams'
wait for each other to start. A barrier's time is the rest divided by the
rounds.

## Building and running

`run.sh [build-dir] [offload-arch]` builds the program and runs both shapes,
over numbers of teams, on the GPU and on the host's teams, which libomp runs as
threads. The build directory defaults to `build-merk`, beside `llvm-project`.
Without an offload-arch, the GPU's measurements are left out.
`TEAM_BARRIER_ROUNDS` sets the rounds, 10000 by default, and
`TEAM_BARRIER_TEAMS` the numbers of teams, `2 4 16 64` by default.

```sh
merkurius/benchmarks/team-barrier/run.sh "" gfx1030
```

`teambarrier central|tree TEAMS ROUNDS` runs one measurement.

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
