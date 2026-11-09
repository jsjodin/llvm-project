# GPU to GPU

Two GPUs, on two nodes, play ping-pong: GPU A sends an array of numbers, GPU B
adds them up and sends back the sum, three times.

- `gpu_to_gpu.m` is the demo. The sends and receives are in `gpu_a` and `gpu_b`.
- `demo_node.m` starts the nodes, connects them and prints the messages.
- `run.py` starts both nodes on one machine.

## Running it

From this directory, with `merkc` built, and for a GPU whose architecture is
`gfx1030`:

    merkc -O2 -fopenmp --offload-arch=gfx1030 gpu_to_gpu.m demo_node.m -o gpu_to_gpu
    OMP_TARGET_OFFLOAD=MANDATORY python3 run.py ./gpu_to_gpu

The linker and the program need to find `libomp`, as for the other GPU programs
(`benchmarks/pingpong/run.sh` shows the paths in a build tree). Without
`-fopenmp`, the regions run on the CPU.

On one machine the two nodes are two operating system processes sharing one GPU,
talking over TCP on loopback.

On two machines, give both the same cookie, and start node B first. It prints
the port it listens on:

    MERK_COOKIE=secret ./gpu_to_gpu b 0.0.0.0:7000
    MERK_COOKIE=secret ./gpu_to_gpu a HOSTNAME_OF_B:7000
