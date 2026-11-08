# Processes

What a Merkurius process costs on one worker. Results are in the design
document, "Measurements".

    merkc -O2 processes.m -o processes
    ./processes pingpong 1000000
    ./processes spawn 1000000 [STACK_BYTES]

`spawn` needs the address space of its stacks: 64 TB for a million processes
with the default stack, so raise any `ulimit -v`.
