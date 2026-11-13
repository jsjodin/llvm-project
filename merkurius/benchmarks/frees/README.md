# A GPU process's blocks

What the blocks of a GPU process cost (docs/INPROGRESS.md, Memory that a
GPU process gives back, f4), beside a message's round trip. Results are in
the design document, "Measurements", "A GPU process's blocks".

`frees.m` spawns a process on the GPU and has it run one measurement's
rounds; the host times them with `clock_gettime` and takes away one round
trip, measured first.

| Name | Each round |
| --- | --- |
| `ping` | a message to the process and its answer (no rounds on the GPU) |
| `alloc` | `merk_device_alloc` and `merk_device_free` of BYTES, from C in the process: the runtime's path for its cells and arrays (`runtime/device/device.cpp`, `NewBlock`) |
| `malloc` | LLVM libc's `malloc` and `free` of BYTES alone: the floor under `alloc` |
| `cold-alloc`, `cold-malloc` | the same, each block 2 MiB bigger than the last, so that no cache has one |
| `array` | `array.init` of WORDS ints, freed at its last use by the compiler (`Frees.cpp`) |
| `none` | the loop alone |

    run.sh [build-dir] [offload-arch]

or by hand:

    merkc -O2 -fprocess-declarations --offload-arch=gfx1030 -L... frees.m -o frees
    ./frees gpu alloc 100000 4096
