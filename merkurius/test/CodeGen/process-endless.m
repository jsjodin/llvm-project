% A GPU process that never ends: the echo answers each number it receives, then
% waits for another with no timeout, until the program exits. LTO inlines its
% receive, as it does beside math.sqrt's code (with only arithmetic it keeps the
% receive a call), and then knows that the dispatch that the processes' kernel's
% RunProcess calls never returns. Only lane 0 of a wave calls RunProcess, while
% the wave's other lanes wait for its slot to change, and the lanes of a wave
% make no progress of their own: RunProcess must keep a way back, or the backend
% may run the other lanes' loop first, which waits for ever (device.cpp,
% RunProcess). process-endless-gpu.test runs the program.
% REQUIRES: amdgpu-registered-target
% RUN: rm -rf %t.dir && mkdir %t.dir
% RUN: merkc -c -O2 -fopenmp -fprocess-declarations --offload-arch=gfx1030 -save-temps=obj %s -o %t.dir/p.o
% RUN: llvm-objdump -d --no-show-raw-insn %t.dir/process-endless-gfx1030-processes.elf | FileCheck %s --check-prefix=RETURNS
% RETURNS-LABEL: <_ZN12_GLOBAL__N_110RunProcess{{[^>]*}}>:
% RETURNS-NOT: {{^[0-9a-f]+ <}}
% RETURNS: s_setpc_b64

:- module process_endless.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, math, process.

    % Answers each number it receives with its square root, for ever.
:- process echo(mailbox(float)) receives float on gpu.

echo(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(X),
        send(Reply, math.sqrt(X), !IO),
        echo(Reply, Self, !IO)
    ;
        R = timeout
    ).

main(!IO) :-
    new_mailbox(Me, !IO),
    spawn(gpu(0), echo(Me), Echo, !IO),
    send(Echo, 2.25, !IO),
    receive(Me, any, infinity, R, !IO),
    ( if R = message(Y) then
        io.write_float(Y, !IO),
        io.nl(!IO)
    else
        true
    ).

% OUTPUT:1.5
