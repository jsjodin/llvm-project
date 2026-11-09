# Stress tests

These are run by hand, not by lit.

## gc_stress.m

Processes each keep a large table of mixed terms with a checksum for each,
replace random entries so that old data dies, check everything they read, and
send entries to each other around a ring. A mismatch or a crash is a failure.
A run ends by the clock, and each process then reports its rounds, what it
received, and a lower bound on how many rounds had a collection.

    merkc -O2 gc_stress.m -o gc_stress
    STRESS_SECONDS=1800 STRESS_WORKERS=8 STRESS_ENTRIES=150000 ./gc_stress

| Variable | Meaning | Default |
| --- | --- | --- |
| `STRESS_SECONDS` | how long to run | 30 |
| `STRESS_WORKERS` | processes | 4 |
| `STRESS_ENTRIES` | entries in each process's table, about 10 KB each | 2000 |
| `STRESS_SEED` | varies the data | 12345 |

The settings above peak at about 48 GB. A small table collects far more often
relative to its work. `MERK_GC_EVERY_ALLOC=N` collects at every N-th allocation.
