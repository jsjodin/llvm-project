# Benchmarks

Programs that measure what Merkurius's code does, rather than test it.
Each has a lit test in `test/Benchmarks` that builds it and runs it on a
small input, so that it keeps working; the benchmarks themselves are run
by hand.

| Benchmark | What it measures |
| --- | --- |
| [`array-messages`](array-messages/README.md) | Bandwidth of arrays as messages, from a GPU's target region or the CPU, over message sizes, beside `omp_target_memcpy` alone |
| [`bandwidth`](bandwidth/README.md) | Memory bandwidth of five array kernels under OpenMP, on the host's threads or a GPU, modeled on BabelStream |
| [`frees`](frees/README.md) | What a GPU process's blocks cost to allocate and free, small and large, beside libc's `malloc` and a message's round trip |
| [`front`](front/README.md) | A reacting front burning through a grid, dense, on one GPU, in Mercury and in C: heavy chemistry where the front is, the demo's main reference |
| [`gpu-pairs`](gpu-pairs/README.md) | Arrays between GPU processes on two GPUs of a node, and on one GPU, through relays: latency and bandwidth of a hop, the baseline for direct sends between GPUs |
| [`gpu-link`](gpu-link/README.md) | Arrays streamed between GPU processes on two GPUs with no host on the way, one way and both: the direct path against the link |
| [`life`](life/README.md) | Conway's Game of Life, dense, on one GPU, in Mercury and in C with cells of 8 and 1 bytes, on two nodes over TCP with halo rows as messages, and on sparse tiles, in one GPU process and on several places with a directory: the demo's references |
| [`mandel-gpus`](mandel-gpus/README.md) | The work-stealing example's Mandelbrot zoom on four GPUs, 16 places: static, stealing in a ring or own GPU first, a host pool and bands, with and without a barrier between frames, beside C with OpenMP; tiles, sizes, places a GPU, traces of single frames |
| [`open-holes`](open-holes/README.md) | A list built through holes bound at every step, then summed as an open term: E5's shape, compiled |
| [`open-unify`](open-unify/README.md) | Two open lists, one with holes, unified by the type's unify procedure, then summed |
| [`pingpong`](pingpong/README.md) | Latency of a message round trip between processes, from a GPU's target region or the CPU, over TCP or within a node, beside plain C floors |
| [`receive-goal`](receive-goal/README.md) | Throughput of receive goals between the host's processes: one mailbox, two, and messages no case selects in front |
| [`team-barrier`](team-barrier/README.md) | Time of a barrier between the teams of a target teams region, from team mailboxes, central or a tree, on a GPU or the host's teams |
