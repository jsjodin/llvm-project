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
