# Benchmarks

Programs that measure what Merkurius's code does, rather than test it.
Each has a lit test in `test/Benchmarks` that builds it and runs it on a
small input, so that it keeps working; the benchmarks themselves are run
by hand.

| Benchmark | What it measures |
| --- | --- |
| [`array-messages`](array-messages/README.md) | Bandwidth of arrays as messages, from a GPU's target region or the CPU, over message sizes, beside `omp_target_memcpy` alone |
| [`bandwidth`](bandwidth/README.md) | Memory bandwidth of five array kernels under OpenMP, on the host's threads or a GPU, modeled on BabelStream |
| [`pingpong`](pingpong/README.md) | Latency of a message round trip between processes, from a GPU's target region or the CPU, over TCP or within a node, beside plain C floors |
