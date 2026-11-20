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

### 2d. Fix 1: merkc passes a flat record a word for each field

merkc now splits a flat record that a loop in a GPU process takes into one
word per field, for every kind of loop (`LoopRecord`,
lib/CodeGen/MLIRGenProcesses.cpp). It takes the record apart with
`merk.deconstruct` before the loop's words, and the rounds make it again
with `merk.construct` for each call of the body. A flat record here has
one constructor, no hidden fields, and at least two fields that are all
integers or floats. Lowering keeps such records by value, so neither side
makes a cell. Other records still go as one boxed word. In the port's
`-O2` process IR, all six 88-byte boxings of `P` are gone. Test:
`test/CodeGen/process-loop-records.m` and `-gpu.test`.

A/B on GPU 4: the unchanged port built by the previous and the new code
generator, alternating, 3 rounds each (kernel 3 old: 1 round), 2048²,
median ms a step. Every pair gave the same checksum.

| Case and kernel | Before | After | After / before |
|---|---|---|---|
| Dam break, kernel 0 | 1.742 | 1.723 | 0.99 |
| Dam break, kernel 2, tiles of 32 | 2.859 | 3.154 | **1.10** |
| Dam break, kernel 2, tiles of 64 | 3.029 | 3.017 | 1.00 |
| Valley, kernel 0 | 1.104 | 1.098 | 0.99 |
| Valley, kernel 2, tiles of 32 | 2.202 | 2.219 | 1.01 |
| Valley, kernel 3, tiles of 32 | 30.597 | **1.569** | 0.05 |

- **Kernel 3 is fixed:** it is now 1.40× faster than kernel 2 on the valley,
  with the same steps and checksum. That matches the hand-edited copy E
  (1.564 ms).
- **The dam break's kernel 2 with tiles of 32 is 10% slower, but the fix
  doesn't cause it.** Copy E only changes `tile_update`'s source, and the
  previous compiler builds it the same way. Built that way, E was just as
  slow: 3.07 ms against 2.87 for the original (4 runs each). That loop's
  generated code is almost the same before and after: `merk_process_loop`'s
  IR differs by 18 `readfirstlane` calls and one alloc/free pair. Splitting
  records only for team loops didn't help either (3.115 ms).
- **What it is:** every loop body of the module is inlined into one function,
  `merk_process_loop`. Changing kernel 3's code shifts the register
  allocation and layout of kernel 2's code, which runs at about 10% different
  speeds with tiles of 32 and 16. Kernel 2 with tiles of 16, whose faces fit
  in LDS, was also slower: 3.78 against 4.40 ms, with more spread. The
  alignment experiment's ±1.5% (2026-11-20, gfx1030) looks like the same
  kind of effect.

Open, found on the way:
- **Fragile code generation:** the loops of a module share one function, so
  any change to one loop can move another's speed by 10%. Each loop's
  cases could be kept apart, for example by not inlining a loop's rounds
  into `merk_process_loop`'s switch. Not tried.
- **Group-private fallback:** a group-private array that doesn't fit in LDS
  falls back to libc's GPU malloc and free, once per iteration
  (`GroupAllocate`, runtime/device/device.cpp). Kernel 2's face array is
  101 KB with tiles of 32 and 400 KB with tiles of 64, against 64 KB of
  LDS, so it pays that every tile in every step. That could be part of why
  kernel 2 loses to kernel 0 on the gfx90a (item 1).
- **Fix 2 (runtime) is still open:** a team's first wave that really makes a
  cell in a distribute body still goes to the GPU's malloc and keeps the
  block until the process ends.

### 2e. SERGHEI against the port, alternating, after fix 1

```
scripts/compare-dambreak.sh gfx90a 2048 5                         # GPU 4
SERGHEI_VARIANT=nofma scripts/compare-valley.sh gfx90a 2048 3     # GPU 6
```

In each round SERGHEI and the port took turns on one GPU. The port was
built by merkc with fix 1 (2d). The dam break ran on GPU 4 and the valley
on GPU 6, at the same time, on different packages. The baseline is
SERGHEI's default build for the dam break, and its no-FMA build for the
valley, whose default build stalls (item 2). Times are medians; the spread
across rounds is at most 1.8%.

| 2048², ms a step | Dam break (252 steps) | vs SERGHEI | Valley (7182 / 7270 steps) | vs SERGHEI |
|---|---|---|---|---|
| SERGHEI (Kokkos, HIP), `swe.solve` | 2.598 | 1.00× | 1.999 | 1.00× |
| SERGHEI, whole run | 15.565 | | 2.972 | |
| Port, kernel 0 | **1.731** | **1.50×** | **1.099** | **1.82×** |
| Port, kernel 2, tiles of 32 | 3.153 | 0.82× | 2.203 | 0.91× |
| Port, kernel 2, tiles of 64 | 3.061 | 0.85× | | |
| Port, kernel 3, tiles of 32 | | | 1.565 | 1.28× |

- **Same results:** all runs of a case end at the same volume. On the
  valley the port takes 7270 steps and SERGHEI-nofma 7182, so the times are
  compared per step.
- **The comparison is against SERGHEI's best case:** `swe.solve` leaves out
  its volume sum each step and its output. Its whole run per step, with
  managed memory (`Kokkos::SharedSpace` for every array, src/define.h),
  is 9.0× kernel 0's on the dam break and 2.7× on the valley.
- **Kernel 0 is the port's best case on the gfx90a.** The tiled kernels lose
  to it. Kernel 3 beats SERGHEI on the valley, but not kernel 0.
- **Agrees with the earlier, separate runs** (items 1 and 2b, the A/B in
  2d) within 2.5%. SERGHEI-nofma on the valley was 2.3% faster here than
  in 2b (1.999 against 2.046 ms), with that run on a different GPU.

Logs: `merkurius-serghei-swe/results/dambreak-gfx90a-2048-20261120-072716`
and `results/valley-gfx90a-2048-20261120-072719` (not in git).

### 3. Register budget on swe-muscl: waves 2, 3 and 4

```
MERKC_FLAGS=-fgpu-process-waves=$w SWE_KERNEL=$k \
  ./compare-swe-muscl.sh $B gfx90a 5          # w 2 3 4, k 1 2, GPU 4
```

swe-muscl at its defaults: a 2048² world, tiles of 32, 1000 steps, at
most 484 tiles. 5 rounds each, Mercury, C (OpenMP offload) and HIP
taking turns. Every program gave the same checksum, 7e6ebd06a6e0a7a6.
Times are the average step in ms; the C and HIP programs don't depend on
the waves, so their times are the same in every run, within 1%.

| Waves | Mercury `step` (k1) | Mercury `step_faces` (k2) | C | HIP k1 | HIP k2 |
|---|---|---|---|---|---|
| 2 (today's default) | **0.304** | **0.556** | 0.228 | 0.310 | 0.250 |
| 3 | 0.361 | 0.745 | 0.228 | 0.310 | 0.248 |
| 4 | 0.573 | 0.757 | 0.227 | 0.310 | 0.248 |

- **2 waves is best for both kernels,** by 19% to 88%. The default stays,
  and by the rule in experiments.txt, ca8b002 (`inreg`) doesn't stay: it
  was only to be kept if 3 or 4 waves beat 2.
- **Kernel 1 matches HIP's kernel 1** (0.304 against 0.310 ms). Mercury's
  bookkeeping, 0.078 ms a step, is included; HIP's is 0.053.
- **C with OpenMP offload is the fastest program** at 0.228 ms. Mercury's
  best is 1.33× slower than it, and HIP's best (k2) 1.09× slower.
- **Mercury's kernel 2 is 2.2× slower than HIP's kernel 2,** and slower
  than Mercury's own kernel 1: the same pattern as the SERGHEI port's
  tiled kernel 2 (items 1, 2e). step_faces' group-private arrays are
  4 × 32 × 33 and 3 × 32 × 32 doubles, 58 KB together. If the team's
  share of LDS is less than that, they fall back to the GPU's malloc in
  every tile (2d, "group-private fallback"). Not checked yet.
- **The goal, Mercury 1.5 to 2× faster than today on the gfx90a, isn't
  reached by the register budget:** more waves only make it slower.

Logs: the per-run traces went to /tmp (deleted); the summary is in this
session's scratchpad, waves.log.

### 3b. Do step_faces' group-private arrays fit in LDS? Not at 2 waves

A team's share of LDS for group-private arrays is `GroupBytes`
(runtime/gpu-processes.cpp): the most dynamic LDS that keeps the
kernel's occupancy. swe-muscl's kernel has 248 VGPRs, so 2 waves a SIMD;
a team of 256 threads is 4 waves of 64, so two teams share a CU's 64 KB.
A temporary probe in `GroupAllocate` (reverted) printed what the runtime
chose, and the first array that didn't fit:

| Build | Team's LDS budget | step_faces `Faces0` (33,800 B) | `Diffs0` (24,584 B) |
|---|---|---|---|
| swe-muscl, 2 waves (default) | 31,232 B | malloc'd in every tile | in LDS |
| swe-muscl, 1 wave | 64,000 B | in LDS | in LDS |
| SERGHEI port, kernel 2, tiles of 32 (101,384 B) | 31,232 B | malloc'd in every tile | |

So at the default, every tile of `step_faces`, and of the port's kernel 2,
pays the GPU's malloc and free for its faces, and keeps them in global
memory.

But that isn't most of kernel 2's gap to HIP (item 3: 0.556 against 0.250 ms):

| swe-muscl, ms a step (3 rounds, GPU 4) | 1 wave | 2 waves (item 3) |
|---|---|---|
| Mercury `step` (k1) | 0.364 | 0.304 |
| Mercury `step_faces` (k2) | 0.550 | 0.556 |
| HIP `step_faces` | 0.250 | 0.250 |

- **Occupancy costs `step` 20% at 1 wave.** If `step_faces` lost the same,
  fitting its arrays in LDS gained about 17%.
- **HIP's `step_faces` runs at the same occupancy as Mercury's 1-wave
  build.** It asks for the 33.8 KB of faces in dynamic LDS, one block a CU,
  so 1 wave a SIMD. Even then, with nothing malloc'd, Mercury's is 2.2×
  slower.
- **HIP keeps each thread's differences in registers** (4 cells a thread,
  `dm`/`dn`/`da`). Mercury keeps them in a second group-private array,
  `Diffs0`, with a team loop and its barriers to write and read them.

Open:
- **Where the rest of the 2.2× goes.** Candidates: `Diffs0` in LDS against
  registers, and the per-tile team loops (`x_faces`, `x_diffs`, ...)
  with their barriers against HIP's one kernel body. A probe of the
  per-phase times would tell.
- **Fallback cost:** a team could keep its fallback block from one tile to
  the next instead of a malloc and free each time. That's cheap to try,
  and helps whenever an array doesn't fit.
- **Choosing the budget:** when a module's group-private arrays need more
  than the occupancy-preserving share, the runtime could size dynamic LDS
  for them and give up occupancy, as HIP's launch does.

### 3c. Where step_faces' time goes, phase by phase, against HIP's

Scratch copies of swe-muscl.m and swe-muscl.hip (not in git) read the
GPU's steady counter (`__builtin_readsteadycounter`, 25 MHz on the gfx90a)
in each tile of `step_faces`, between its phases. Mercury reads it in the
distribute body, by the team's first wave, between the team loops; HIP
reads it in every thread after each `__syncthreads`, and thread 0 adds the
times up. Both runs are at the defaults (2048², tiles of 32, 1000 steps,
253,090 tile-steps), on GPU 4, with the usual checksum. Two runs of each
agree within 1%.

| µs per tile | Mercury, 2 waves | Mercury, 1 wave | HIP (1 block a CU) | Mercury 1 wave / HIP |
|---|---|---|---|---|
| Group-private arrays (`GroupAllocate`) | 27.9 | 6.8 | | |
| `x_faces` (1056 faces) | 89.8 | 61.6 | 24.7 | 2.5× |
| `x_diffs` | 19.8 | 6.1 | 0.7 | 8.4× |
| `y_faces` (1056 faces) | 74.2 | 52.6 | 23.5 | 2.2× |
| `faces_tile` (1024 cells) | 56.1 | 32.5 | 11.4 | 2.8× |
| **Total** | **268** | **160** | **60** | **2.7×** |
| Whole step, ms (host) | 0.571 | 0.583 | 0.252 | |

- **No single phase explains the gap.** The arithmetic-heavy phases,
  the two face loops and the cell update, each take 2.2 to 2.8× HIP's
  time at the same occupancy. Together they are 92% of the 1-wave tile.
- **At 2 waves, the malloc fallback costs 28 µs a tile** (3b), 10% of the
  tile, and two teams on each CU make every phase about 1.5× slower than at
  1 wave. The step's time stays the same, because twice as many tiles run
  at once.
- **`x_diffs` is a team loop where HIP has none:** 6 to 20 µs, against
  0.7 µs for HIP's loads into registers.
- **The two programs have the same structure.** HIP's `step_faces` also
  walks its 1056 faces in 5 rounds of 256 threads, with the same
  `faces_at` and `hllc` calls, and a barrier after each phase.

Differences that could account for the 2.2 to 2.8×, none tested yet:
- **Registers:** HIP's `step_faces` has 120 VGPRs and no scratch.
  Mercury's process kernel has 248 VGPRs and 768 bytes of scratch a lane,
  for all the module's loops in one `merk_process_loop` (2d).
- **Barriers:** Mercury's team loops have a barrier after every round
  (`merk_device_sync_threads` in `ProcessLoopRounds`), 5 per face loop.
  HIP has one after the loop. INPROGRESS.md (d2) measured the per-round
  barrier as a gain of 8 to 17% on light loops; this body is heavy.
- **Calls:** each round calls the body, and the rounds are recursion turned
  into a loop. HIP's loop body is inlined into the kernel.

### 3d. Without the per-round barrier: no gain

A temporary switch in `ProcessLoopRounds` (reverted) made merkc keep a
team loop's barrier only after its last round, not after every round.
The 3c phase-timed build ran with and without it, two runs each, on GPU 4.
Every run gave the usual checksum.

| µs per tile | 2 waves, per round | 2 waves, last only | 1 wave, per round | 1 wave, last only |
|---|---|---|---|---|
| Group-private arrays | 27.8 | 28.6 | 7.1 | 6.3 |
| `x_faces` | 89.7 | 87.0 | 63.1 | 62.9 |
| `x_diffs` | 19.8 | 24.0 | 6.1 | 5.4 |
| `y_faces` | 74.1 | 69.4 | 52.0 | 49.6 |
| `faces_tile` | 56.3 | 56.5 | 33.0 | 31.9 |
| Whole step, ms (host) | 0.571 | 0.564 | 0.588 | 0.571 |

- **The barriers cost 1 to 3% of the step,** within the noise of a phase
  or two. They aren't the 2.7× gap to HIP (3c), so the per-round barrier
  stays, as INPROGRESS.md (d2) found it a gain on light loops.
- **One run of the 1-wave build without the barriers printed nothing.**
  Its output wasn't kept. 31 more runs of the variants without the barriers
  all passed, so it may be the switch, or a flake like gpu-pairs'.
  Not followed up, since the switch was reverted.

Left from 3c: registers and scratch (248 VGPRs and 768 B against HIP's
120 and none), and the body called from the rounds rather than inlined.

### 3e. Registers, scratch and code: not the gap either

From the assembly of swe-muscl's process code, `-save-temps`, at the
default 2 waves:

| Function | VGPRs | Scratch, B a lane | Code |
|---|---|---|---|
| `merk_team_loop` (every loop under parallel do in a distribute body) | 248 | 304 | 60 KB |
| `merk_process_loop` | 248 | 576 | 39 KB |
| HIP `step_faces` | 120 | 0 | |

- **The bodies are inlined** into `merk_team_loop`: its only calls are
  to `merk_device_uncaught_exception` and `memcpy`.
- **Loop by loop, the code is HIP's size.** Matched by their
  `v_div_scale_f64`s (60 in a face loop, 26 in the cell update):

  | Loop | Mercury instructions | HIP instructions | Mercury spill ops |
  |---|---|---|---|
  | face loop | 1698 | 1720 (`x_faces`) | 0 |
  | face loop | 1821 | 1410 (`y_faces`) | 50 |
  | cell update | 965 | 905 (`faces_tile`) | 0 |

  Spills are in one face loop and in one small loop. Both face loops are
  2.2 to 2.5× HIP's time (3c), with or without spills.
- **Every array access is flat** in Mercury (`flat_load`, `flat_store`):
  global arrays and the group-private ones in LDS alike. HIP uses `global_*`
  and `ds_*`. Forcing flat on HIP's `step_faces`, by hiding its pointers
  behind an empty `asm`, keeps it at 120 VGPRs and costs it 7%
  (0.251 → 0.266 ms a step; `x_diffs` 0.7 → 1.2 µs a tile). Flat is a
  small part of the gap, and not what raises Mercury's VGPRs.

So the same instructions take 2.5× as long. They aren't spills or a lot
more code. They're stalls, or time outside the loop bodies:
- **A fixed cost per team loop** is likely: `x_diffs`, a few loads a
  thread, takes 6.1 µs a tile at 1 wave against HIP's 0.7. That suggests
  roughly 5 µs to start and end each team loop (the first wave writing its
  words, the other waves noticing, the join), about 20 µs of the 4 loops'
  100 µs gap.
- **The rest** needs the hardware's counters, but rocprofv3 records
  nothing from Mercury's persistent `merk_place` kernel (it does for HIP).
  A probe would have to read them on the device, or time a team loop's
  start, rounds and join separately.

### 3f. A team loop's start, rounds and join

A temporary probe in the device runtime (reverted) timed each team loop
from `merk_device_team_loop`, in the first wave's first lane:
- **start:** the barrier that wakes the other waves;
- **rounds:** the first wave's `merk_team_loop`;
- **reduce:** the reductions across the team;
- **join:** the last barrier.

The probe ran on the 3c phase-timed build, two runs each, which agree
within 2%. The probe itself makes the phases up to 25% slower (`x_diffs`
6.1 → 8.6 µs a tile at 1 wave).

| µs per call | 1 wave: start | rounds | reduce | join | 2 waves: start | rounds | reduce | join |
|---|---|---|---|---|---|---|---|---|
| `x_faces` (loop 3) | 0.05 | 60.2 | 0.1 | 0.7 | 0.05 | 82.5 | 0.1 | 1.7 |
| `y_faces` (loop 4) | 0.05 | 53.5 | 0.1 | 0.7 | 0.05 | 71.8 | 0.1 | 1.5 |
| `x_diffs` (loop 5) | 0.05 | 5.3 | 0.1 | 0.5 | 0.07 | 19.6 | 0.1 | 1.7 |
| `faces_tile` (loop 6) | 0.06 | 30.2 | 5.6 | 1.1 | 0.10 | 47.6 | 7.7 | 1.2 |

- **Starting and joining a team loop costs under 2 µs.** The guess in 3e,
  about 5 µs a loop, is wrong: the time is in the rounds.
- **At 1 wave every group-private array is in LDS.** A second probe
  counted 400,079 arrays that fit and none that fell back. At 2 waves,
  half fell back (199,988 fit, 200,195 malloc'd: `Faces0` each time).
- **`x_diffs`' rounds are slow even in LDS.** At 1 wave, 4 rounds take
  5.3 µs, about 1.3 µs a round. Each round of its code is 4 flat loads,
  3 adds, 3 flat stores, an `s_waitcnt lgkmcnt(0)` and a barrier. HIP's
  four loads into registers take 0.7 µs in all, or 1.2 µs with every
  access forced flat (3e). Without the per-round barrier, it's still
  5.4 µs (3d).
- **`faces_tile`'s reductions take 5.6 µs** at 1 wave (`ReduceAcrossTeam`,
  3 reductions). HIP's `reduce` is inside its 11.4 µs.

Open: why a round of a few flat accesses to LDS takes about a
microsecond in a team loop. A microbenchmark would show it: a team loop
over a group-private array, against the same loads in HIP, flat and `ds_`.

### 3g. LDS microbenchmark: rounds are cheap, a team loop costs ~1.5 µs

Scratch programs (not in git): a Mercury process and a HIP twin. Each tile
runs four loops across a team of 256 threads, timed by the steady counter:
- **fill:** `G[k] := k`, into LDS;
- **one store:** `D[k] := 0`, into LDS;
- **diffs:** `x_diffs`' body, LDS to LDS;
- **diffs, global:** the same body on global arrays.

The HIP twin comes in four builds: `ds_` or flat addressing (pointers
hidden behind an empty `asm`), each with HIP's own loops or with
Mercury's rounds (a guarded iteration and a `__syncthreads` in each round).

**Not a kernel setting.** `merk_place`'s descriptor matches HIP's apart
from scratch, with no tgsplit and the same mode bits. rocm-smi shows both
at 1700 MHz throughout a long run.

**The first run, with tiles of 32, wasn't a fair comparison.** The
microbenchmark's kernel has 185 VGPRs (`-fgpu-process-waves=1` only caps
them), so two teams fit per CU. Each team got 31 KB of LDS, so `G`
(33.8 KB) fell back to global memory, and two tiles shared each CU where
HIP ran one.

**At matched occupancy:** tiles of 16, 1760 of them, two teams per CU.
Every array fits in Mercury's 31,232 bytes, and HIP asks for 31,232 bytes
of dynamic LDS. Two runs each, alternating, within 1%:

| µs per tile | fill (5 rounds) | one store (3) | diffs, LDS (1) | diffs, global (1) |
|---|---|---|---|---|
| Mercury | 2.01 | 1.58 | 2.30 | 3.04 |
| HIP, `ds_`, Mercury's rounds | 0.35 | 0.22 | 0.33 | 2.00 |
| HIP, flat, Mercury's rounds | 0.48 | 0.31 | 0.56 | 1.97 |
| HIP, `ds_`, its own loops | 0.29 | 0.19 | 0.39 | 2.02 |

- **Mercury adds 1.1 to 1.7 µs to each team loop,** whether it has 1
  round or 5. So the rounds are cheap, and the cost is per loop: the
  first wave writing the loop's words, the start, the join, and the calls
  into `merk_team_loop`. 3f timed start and join from inside at under
  2 µs.
- **Flat costs HIP 0.1 to 0.2 µs a loop here,** and Mercury's round
  structure about as much.

**This contradicts 3f.** There, swe-muscl's `x_diffs` spent 5.3 µs in 4
rounds at one team per CU, with its arrays in LDS. Here a round costs
next to nothing. The obvious difference is the code around the loop: in
swe-muscl, `merk_team_loop` is 60 KB, with 248 VGPRs and 304 bytes of
scratch; here it's small. To test that, grow this microbenchmark's
`merk_team_loop`, for example with swe-muscl's face loops as further
cases that never run, and see whether the rounds slow down (instruction
cache, or spills on entry).

### 3h. The cost is merk_team_loop's callee-saved registers

The 3g microbenchmark's loops went into a copy of swe-muscl.m (scratch, not
in git), as a further GPU process that `main` runs under `SWE_BENCH=1`. The
module's loops all share one `merk_team_loop`, so the microbenchmark's
loops run inside swe-muscl's 63 KB function, and swe-muscl's own loops
never run. Both builds run two teams per CU, and the arrays fit in LDS.

| µs per tile, tiles of 16 | `merk_team_loop` | fill (5 rounds) | one store (3) | diffs, LDS (1) | diffs, global (1) |
|---|---|---|---|---|---|
| microbenchmark alone | 2.7 KB, 40 VGPRs, 96 B scratch | 2.02 | 1.58 | 2.29 | 3.03 |
| inside swe-muscl | 63 KB, 248 VGPRs, 304 B scratch | 21.1 | 18.0 | 20.5 | 22.9 |
| inside swe-muscl, `-fgpu-process-no-callee-saved` | 61 KB, 197 VGPRs, 112 B scratch | 2.35 | 1.97 | 2.64 | 3.34 |

- **Inside swe-muscl, every team loop costs about 18 µs more,** whatever
  its rounds. That is paid on each call of `merk_team_loop`, whose
  prologue saves 50 VGPRs to scratch (`buffer_store_dword`) and 37 SGPRs
  into lanes, and whose epilogue restores them. It explains 3f's
  `x_diffs`, 19.6 µs at 2 waves and 5.3 at 1, nearly all of it outside
  the work.
- **Without callee-saved registers, nearly all of it goes:** within
  0.3 µs a loop of the microbenchmark alone.

swe-muscl itself, `compare-swe-muscl.sh` 3 rounds, GPU 4, same checksum
(this is item 4's first configuration):

| ms a step | Mercury, default | Mercury, no callee-saved | C | HIP |
|---|---|---|---|---|
| `step` (k1) | 0.304 | **0.232** (1.31×) | 0.235 | 0.311 |
| `step_faces` (k2) | 0.555 | **0.394** (1.41×) | 0.234 | 0.252 |

- **Kernel 1 now beats HIP** (0.232 against 0.311 ms) and ties C with
  OpenMP offload.
- **Kernel 2 is still 1.56× HIP.** What's left includes the malloc
  fallback for `Faces0` at 2 waves (3b, 28 µs a tile), about 2 µs a team
  loop (3g), and `x_diffs` as a loop of its own.
- **By item 4's rule this is a clear gain,** so 4745adb and 2b5a81e stay.
  The flag is a candidate for the gfx90a's default. The rest of item 4
  (with 3 waves, and `merk_process_loop`'s side) still has to run.

### 4. No callee-saved registers (-fgpu-process-no-callee-saved)

**swe-muscl** (`compare-swe-muscl.sh` 5 rounds, GPU 4, same checksum). The
columns without the flag are item 3's:

| ms a step | 2 waves | 2 waves, no callee-saved | 3 waves | 3 waves, no callee-saved |
|---|---|---|---|---|
| `step` (k1) | 0.304 | **0.231** | 0.361 | 0.323 |
| `step_faces` (k2) | 0.556 | **0.394** | 0.745 | 0.645 |
| C | 0.228 | 0.235 | 0.228 | 0.241 |
| HIP (k1 / k2) | 0.310 / 0.250 | 0.311 / 0.249 | 0.310 / 0.248 | 0.311 / 0.251 |

**The SERGHEI port** with the flag, against 2e's runs without it. Dam
break: 5 rounds on GPU 6. Valley: 3 rounds on GPU 2, against
SERGHEI-nofma. Both ran at the same time and give 2e's volumes.
SERGHEI's own times are within 0.4% of 2e's.

| 2048², ms a step | Dam break, 2e | with the flag | vs SERGHEI | Valley, 2e | with the flag | vs SERGHEI |
|---|---|---|---|---|---|---|
| SERGHEI, `swe.solve` | 2.598 | 2.594 | 1.00× | 1.999 | 1.992 | 1.00× |
| Port, kernel 0 | 1.731 | 1.732 | 1.50× | 1.099 | 1.094 | 1.82× |
| Port, kernel 2, tiles of 32 | 3.153 | **2.839** | 0.91× | 2.203 | **2.012** | 0.99× |
| Port, kernel 2, tiles of 64 | 3.061 | **2.976** | 0.87× | | | |
| Port, kernel 3, tiles of 32 | | | | 1.565 | **1.413** | 1.41× |

- **A clear gain wherever there are team loops:** swe-muscl's kernels
  24 and 29% faster at 2 waves, the port's tiled kernels 3 to 10%.
  Kernel 0, which has no team loops, doesn't change. By the list's rule,
  4745adb and 2b5a81e stay.
- **2 waves stays the default.** At 3 waves the flag helps too, 11 to
  13%, but still loses to 2 waves.
- **The flag should be the gfx90a's default:** nothing measured here
  gets slower. Kernel 0 and SERGHEI's other loops run in
  `merk_process_loop`, which the flag also compiles without callee-saved
  registers, and kernel 0 stays the same.
- **The port's tiled kernels still lose to kernel 0,** and kernel 2 to
  SERGHEI on the dam break. The flag takes the dam break's kernel 2 with
  tiles of 32 back to where it was before fix 1 (2.86, 2d). That fits
  2d's suspicion that fix 1's slowdown there was a side effect of the
  shared function's code.

To pass the flag, merkurius-serghei-swe's compare scripts now take
`MERKC_FLAGS`, as swe-muscl's does (uncommitted there).

Logs: `merkurius-serghei-swe/results/dambreak-gfx90a-2048-20261120-083323`
and `results/valley-gfx90a-2048-20261120-083323`; swe-muscl's in this
session's scratchpad, ncs.log.

### 5. MC E1.1 on the gfx90a: the place against Quicksilver

```
OMPI_CXX=amdclang++ scripts/build_quicksilver.sh hip
QS_ENV="HSA_XNACK=1" GPU_WRAP="$W --gpus 4 --" bench/e11.sh 16 10   # and 32
```

merkurius-mc at 5b6e93e. Quicksilver is LLNL's at eb68bb8 with the P4
patch, built for gfx90a as `bin/qs-hip`. The problem is P1 (CTS2),
10 particles a cell, one rank, on GPU 4. Figures of merit are segments
per second of tracking. Each is a single run.

| E1.1 | 16³ | 32³ |
|---|---|---|
| Place (Mercury GPU process), default | 1.462e7 | 3.516e7 |
| Place, `-fgpu-process-no-callee-saved` | 1.435e7 | 3.417e7 |
| Quicksilver (qs-hip, `HSA_XNACK=1`) | 1.288e6 | 3.606e6 |
| **Place / Quicksilver** | **11.4×** | **9.75×** |

- **The place is about 10× Quicksilver's HIP build on the gfx90a,**
  against 4.77× on the gfx1030, where qs-hip was no speed reference. All
  of Quicksilver's checks pass: absorption, fission and scatter ratios,
  no particles lost, and homogeneous fluence.
- **Quicksilver runs 100 cycles, not 10.** `-N 10` doesn't override
  `nSteps: 100` in `problems/P1_cts2.inp`. Its figure is still an average
  over cycles, but a 32³ run takes about 14 minutes, at 8.3 s of tracking
  a cycle.
- **Without callee-saved registers the place is 2 to 3% slower,** within
  what one run each can tell. The place's tracking runs in the process
  itself, not in team loops, which is where item 4 gained.
- **`e11.sh` raced with `with-gpus.py`:** started right after the place,
  Quicksilver found GPU 4 still held by the place's exiting process
  ("GPU 4 is in use"). `set -e` then ended the script before its summary.
  This happened in both 16³ runs. The 16³ Quicksilver figure above is
  from a separate run of the same command line.

Logs: `merkurius-mc/mc-transport/results/E1.1/20261120-0839-16`,
`-0841-32`, `-0857-16` and `-0859-32` (not in git); Quicksilver at 16³
in this session's scratchpad, qs16.log.

**After fixing `e11.sh`** (merkurius-mc): Quicksilver now runs from a copy
of P1's input with `nSteps` set to the script's cycles. The script waits
for the wrapper to accept a command before starting Quicksilver, and
names its output folders to the second. Both sizes, end to end, at
10 cycles each:

| E1.1, 10 cycles each | 16³ | 32³ |
|---|---|---|
| Place, default | 1.457e7 | 3.491e7 |
| Quicksilver | 1.262e6 | 3.701e6 |
| **Place / Quicksilver** | **11.55×** | **9.43×** |

Quicksilver's figure at 10 cycles is within 3% of its figure at 100.
Logs: `results/E1.1/20261120-090946-16` and `-091106-32`.
