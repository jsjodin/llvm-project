# Experiment results

Results of the runs listed in `experiments.txt`, as they come in. Each entry
says what was run, on what, and what it shows. A result moves to
`INPROGRESS.md` once it leads to an action.

## r11 (gfx90a), 2026-11-20

Setup:
- llvm-project at 987ff6231755 (`merkurius/spills`), with `build-merk` rebuilt
  from it.
- `merkurius-serghei-swe` at 4712169.
- SERGHEI at 2728883, built by `scripts/build-serghei.sh gfx90a`.
- One MI250X die (GPU 4) through `with-gpus.py --gpus 4`. Another user's jobs
  were on GPUs 0–3, and 4–7 were idle.

### 1. SERGHEI dam break: faces-once on the gfx90a

```
scripts/compare-dambreak.sh gfx90a 2048 5
```

Circular dam break, 2048², 8 s, 5 rounds. Times are medians of the average
step.

| Program | ms per step | vs SERGHEI |
|---|---|---|
| SERGHEI (Kokkos, HIP), `swe.solve` timer | 2.605 | 1.00× |
| Port, kernel 0 (each face's solver twice) | 1.746 | 1.49× |
| Port, kernel 2, tiles of 32 | 2.832 | 0.92× |
| Port, kernel 2, tiles of 64 | 3.068 | 0.85× |

- **Results match:** all four take 252 steps to volume 1.443898750e+06 m³.
- **Spread across rounds:** the port's flux time varied by 1–3%. Kernel 0 took
  1.304–1.306 ms, kernel 2/32 2.36–2.42 ms and kernel 2/64 2.57–2.66 ms.
- **SERGHEI's whole run** is 15.905 ms per step. Its `swe.solve` timer, used in
  the table, leaves out the volume sum each step and its output, so the table
  shows SERGHEI's best case.
- **Faces-once reverses on the gfx90a:** on the gfx1030 kernel 2 was 1.64×
  faster than kernel 0. Here kernel 0 is 1.62× faster than kernel 2 with tiles
  of 32.
- **The port is ahead of SERGHEI:** kernel 0 beats SERGHEI's best case by
  1.49×.

Open:
- Why kernel 2 loses here. Not checked yet. One guess is that its state per
  tile costs registers or LDS, and with fp64 and wave64 that cuts occupancy.
- The compiler work in INPROGRESS.md, "Each shared call once a tile, by the
  compiler", was waiting on this run. This result undercuts the reason for it
  on the gfx90a.

Logs: `merkurius-serghei-swe/results/dambreak-gfx90a-2048-20261120-055751`
(not in git).

### 2. SERGHEI valley flood: stopped, SERGHEI stalls

```
scripts/compare-valley.sh gfx90a 2048 3
```

Valley flood, 2048², 600 s; 527,864 of the cells (12.6%) start wet. No
numbers yet: the first run, SERGHEI's (default build, with FMA), never
finished, so the run was stopped.

- **SERGHEI's time step went to exactly 0** between iterations 6000 and
  7000. Simulated time froze at 470.269601950454 s.
- **The first 6000 steps were normal:** dt fell steadily from 0.089 s to
  0.064 s.
- **It then ran 125,000 more iterations** without moving, about 5 minutes
  in all, and was killed.
- **The port didn't run:** the script runs SERGHEI first.
- **Not checked yet:** why SERGHEI's step hits 0 here. On the gfx1030 the port
  matched SERGHEI's no-FMA build (`SERGHEI_VARIANT=nofma`), and the two
  builds take different steps on this case.

Logs: `merkurius-serghei-swe/results/valley-gfx90a-2048-20261120-060800`
(not in git).

### 2b. Valley flood: the port alone, and SERGHEI without FMA

```
MERK_CONFIGS="SWE_KERNEL=0;SWE_KERNEL=2 SWE_TILE=32;SWE_KERNEL=3 SWE_TILE=32" \
  scripts/time-port.sh gfx90a 3 2048 3                             # GPU 4
SERGHEI_VARIANT=nofma SERGHEI_CMAKE_FLAGS=-DCMAKE_CXX_FLAGS=-ffp-contract=off \
  scripts/build-serghei.sh gfx90a
HSA_XNACK=1 build/gfx90a-nofma/bin/serghei build/valley-2048/ <out>/ 1   # GPU 6
```

These runs took the place of item 2's script. The port ran alone, and SERGHEI
without FMA ran once on another package. The port run was stopped after round
2's kernel 2, because each kernel-3 run took 4 minutes. The times below are
from round 1. Round 2 gave the same steps and checksum, and kernels 0 and 2
were within 0.3% of round 1.

| Program | Steps | ms per step | Fluxes only | vs SERGHEI-nofma |
|---|---|---|---|---|
| SERGHEI-nofma, `swe.solve` | 7182 | 2.046 | | 1.00× |
| SERGHEI-nofma, whole run | 7182 | 3.056 | | |
| Port, kernel 0 | 7270 | 1.105 | 0.796 | 1.85× |
| Port, kernel 2, tiles of 32 | 7270 | 2.187 | 1.858 | 0.94× |
| Port, kernel 3, tiles of 32 | 7270 | **32.409** | 1.167 | 0.06× |

- **SERGHEI without FMA finishes:** it reaches 600 s in 7182 steps, at the
  same volume as the port, 1.265744941e+08 m³. Its default build stalls at
  470 s (item 2), so FMA rounding is the likely trigger.
- **Steps differ by 1.2%:** the port takes 7270 steps and SERGHEI-nofma 7182.
  That's the cube-root ulp difference already known from the gfx1030, so the
  times are compared per step.
- **Kernel 0 beats SERGHEI-nofma's best case by 1.85×.**
- **Kernel 2 again loses to kernel 0,** by 1.98× here, as in the dam break.
- **Kernel 3 is broken on the gfx90a:** its fluxes take 1.167 ms a step,
  which is 37% less than kernel 2's, over 43.2% of the 4096 tiles. But its
  whole step takes 32.4 ms. The other 31 ms is spent outside the fluxes,
  probably in the bookkeeping for the active tiles. On the gfx1030, kernel
  3's whole step was 5.26 ms against kernel 2's 6.18 ms.

Open:
- ~~Where kernel 3's 31 ms a step goes.~~ Found, see 2c.
- Why kernels 2 and 3 lose to kernel 0 on the gfx90a (item 1).

Logs: `merkurius-serghei-swe/results/port-gfx90a-3-2048-*` and
`results/valley-nofma-gfx90a-2048-20261120-061636` (not in git).

### 2c. Kernel 3's lost time: a GPU malloc for every tile

Scratch copies of the port, timed with `MERK_SOURCE=<copy> scripts/time-port.sh
gfx90a 3 <N> 1` (builds at -O2), one each on GPUs 4–7. The copies are in the
session scratchpad, not in git.

**Where:** with the step's timer moved to after `update_tiles`, it covered
32.58 of the 32.61 ms. So `update_tiles` takes about 31.4 ms a step, and
`activate` about 0.03 ms.

**Not the reductions:** at 1024², with the wet flag's integer `max` changed
to a float `max`, a step took 7.60 ms, against 7.78 ms for the original.
With that reduction removed, every tile ran and a step took 16.6 ms. The
cost grows with the number of tiles that run.

**Cause:** `tile_update`'s inner `parallel do` calls `cell(P, ...)`, so its
body captures the parameter record `P`: 11 words, 88 bytes. merkc boxes the
record with `merk_device_alloc` to pass it to the team loop, once per tile.
`tile_faces` and `tile_cells` pass scalars and don't allocate. The `-O2` IR
keeps the call. In a worker team's first wave, `merk_device_alloc` goes to
`NewBlock` (runtime/device/device.cpp), whose costs are:
- **GPU malloc:** libc's `malloc` on the GPU, through `RawAllocate`, which
  the runtime's own comment puts at some microseconds each. All the teams
  call it at once.
- **One shared list:** every block is pushed onto it with a compare-and-swap.
- **Never freed:** `merk_device_free` does nothing for a team block until the
  process ends. That's about 1770 blocks a step at 2048², about 13 million in
  a run of 7270 steps.

Kernel 2's flat `update` also boxes `P`, but once per step, in the
process's own wave, from its cache of small blocks.

**Check (copy E):** `cell` takes `n`, `dx` and `hmin` as scalars inside
`tile_update`. That's a measurement copy only, not a fix. Its tile_update
has no allocation in the IR. Steps and checksums are the same as the
original's.

| Valley flood, step (ms) | 1024² | 2048² |
|---|---|---|
| Kernel 0 | | 1.105 |
| Kernel 2, tiles of 32 | 0.730 | 2.187 |
| Kernel 3, original | 7.775 | 32.409 |
| Kernel 3, copy E | **0.603** | **1.564** |

- **Kernel 3 works on the gfx90a without the allocation:** it is 1.21× faster
  than kernel 2 at 1024² and 1.40× at 2048², against 1.15× at 2048² on the
  gfx1030.
- **It still loses to kernel 0:** 1.564 ms against 1.105 ms at 2048². That's
  item 1's question about the tiled kernels.

Fixes to choose from (not done):
1. **merkc:** pass a record that fits in the loop's words (32) by value, as
   scalars, instead of boxing it. That removes the allocation for any
   loop that captures a small record.
2. **Runtime:** make a team's first-wave cells cheap and reclaimed. For
   example, bump them from the team's own region and reset it each
   iteration, as group-private arrays are. That also stops the growth of
   memory for the process's lifetime.

Both are worth doing. 1 removes the cost here, and 2 covers the cases 1
can't, such as a record too large for the loop's words, or a cell that's
really made in the body.
