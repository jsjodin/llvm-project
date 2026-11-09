# GPU to GPU

Two GPUs, on two nodes, play ping-pong: GPU A sends an array of numbers, GPU B
adds them up and sends back the sum, three times.

- `gpu_to_gpu.m` is the demo. The sends and receives are in `gpu_a` and `gpu_b`.
- `demo_node.m` starts the nodes, connects them and prints the messages.

## Running it

From `examples/`, with `merkc` built:

    make run-gpu-to-gpu

That compiles the demo and starts both nodes on this machine. `ARCH=` sets the
GPU's architecture, `gfx1030` by default, and `PORT=` the port. For a machine
without a GPU, `make run-gpu-to-gpu-cpu` runs the regions on the CPU.

On one machine the two nodes are two operating system processes sharing one GPU,
talking over TCP on loopback.

On two machines, build it with `make gpu-to-gpu`, and give both nodes the same
cookie. Node A waits a few seconds for node B to listen:

    MERK_COOKIE=secret OMP_TARGET_OFFLOAD=MANDATORY build/gpu_to_gpu b 0.0.0.0:7000
    MERK_COOKIE=secret OMP_TARGET_OFFLOAD=MANDATORY build/gpu_to_gpu a HOSTNAME_OF_B:7000
