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
| [`life`](life/README.md) | Conway's Game of Life, dense, on one GPU, in Mercury and in C with cells of 8 and 1 bytes, and on two nodes over TCP with halo rows as messages: the demo's first references |
| [`pingpong`](pingpong/README.md) | Latency of a message round trip between processes, from a GPU's target region or the CPU, over TCP or within a node, beside plain C floors |
| [`team-barrier`](team-barrier/README.md) | Time of a barrier between the teams of a target teams region, from team mailboxes, central or a tree, on a GPU or the host's teams |
