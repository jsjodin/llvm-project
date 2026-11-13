% Loops that a GPU process runs across teams, under distribute parallel do
% (docs/INPROGRESS.md, GPU processes that run loops). A worker answers each
% request of an array: it doubles the array's elements into a new array, in
% one loop; adds to each element of that array the one before it in the
% given array, in a second, through a predicate of the module; counts down
% into a third by a negative step; scales an array of floats; and runs a
% loop with no iterations. Large arrays spread the iterations over many
% teams. A process that runs no loops, an echo, runs beside it. The program
% runs its kinds at the host here, where the loops are parallel do loops, or
% sequential without -fopenmp, and at the GPU in process-loops-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% The device's compile gives each loop a number, calls the runtime with it,
% and makes merk_process_loop, which runs a thread's rounds of a loop by its
% number, with the loop's words, with no reductions here
% (process-loop-reductions.m); the host's registers which kinds run
% loops, worker 1 and echo 0. Every function of the device's compile is
% convergent, as a loop's rounds end with a barrier across the team, and the
% kinds' code and the loops stay out of line, so that the registers each
% needs count only while it runs.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -fc1 -fopenmp -fdo-loops -fprocess-declarations -emit-merk -triple x86_64-unknown-linux-gnu -fembed-process-image=gfx1030:%s %s -o - | FileCheck %s --check-prefix=HOST
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t.sequential
% RUN: %libomp-env %t.sequential host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t.sequential2
% RUN: %libomp-env %t.sequential2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_loops.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process.

:- type answer
    --->    answer(int, int, int, int, int, float).

    % Answers each array with the checks of what its loops made, until an
    % empty one.
:- process worker(mailbox(int)) receives array(int) on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(A), array.size(A) > 0 then
        N = array.size(A),
        array.init(N, 0, D0),
        %$omp distribute parallel do
        ( for(I, 0, N - 1), fromto(D0, D1, D2, D) do
            array.set(I, array.lookup(A, I) * 2, D1, D2)
        ),
        add_previous(A, D, E),
        array.init(N, 0, C0),
        %$omp distribute parallel do
        ( for(I, N - 1, 0, -1), fromto(C0, C1, C2, C) do
            array.set(I, N - I, C1, C2)
        ),
        array.init(N, 1.0, F0),
        %$omp distribute parallel do
        ( for(I, 0, N - 1), fromto(F0, F1, F2, F) do
            array.set(I, float(I) * 0.5, F1, F2)
        ),
        %$omp distribute parallel do
        ( for(I, 1, 0), fromto(C, G1, G2, G) do
            array.set(I, -1, G1, G2)
        ),
        send(Reply, sum(D, 0, 0), !IO),
        send(Reply, sum(E, 0, 0), !IO),
        send(Reply, sum(G, 0, 0), !IO),
        send(Reply, truncate_to_int(fsum(F, 0, 0.0)), !IO),
        worker(Reply, Self, !IO)
    else
        true
    ).

    % Each element of D plus the one before it in A, the first plus 0.
:- pred add_previous(array(int)::in, array(int)::in, array(int)::uo) is det.

add_previous(A, D, E) :-
    N = array.size(A),
    array.init(N, 0, E0),
    %$omp distribute parallel do
    ( for(I, 0, N - 1), fromto(E0, E1, E2, E) do
        Previous = ( if I > 0 then array.lookup(A, I - 1) else 0 ),
        array.set(I, array.lookup(D, I) + Previous, E1, E2)
    ).

:- func sum(array(int), int, int) = int.

sum(A, I, S) =
    ( if I < array.size(A) then sum(A, I + 1, S + array.lookup(A, I))
    else S ).

:- func fsum(array(float), int, float) = float.

fsum(A, I, S) =
    ( if I < array.size(A) then fsum(A, I + 1, S + array.lookup(A, I))
    else S ).

    % Answers each number with itself, until 0.
:- process echo(mailbox(int)) receives int on gpu.

echo(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N \= 0 then
        send(Reply, N, !IO),
        echo(Reply, Self, !IO)
    else
        true
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = ["gpu-second"] then
        % A second worker finds every free team the first's.
        new_mailbox(Me, !IO),
        spawn(gpu(0), worker(Me), _, !IO),
        spawn(gpu(0), worker(Me), _, !IO)
    else
        Place = ( if Args = ["gpu"] then gpu(0) else host ),
        new_mailbox(Me, !IO),
        spawn(Place, echo(Me), Echo, !IO),
        spawn(Place, worker(Me), W, !IO),
        list.foldl(ask(W, Echo, Me), [1, 2, 10, 1000, 100000, 1000000], !IO),
        send(W, array([]), !IO),
        send(Echo, 0, !IO)
    ).

    % Sends 1 .. N and prints the answers: the sums of the doubled array,
    % N * (N + 1), of it with the elements before added, N * N + N * (N - 1)
    % / 2 ... written out, of the count down, N * (N + 1) / 2, and of the
    % floats, N * (N - 1) / 4, truncated.
:- pred ask(mailbox(array(int))::in, mailbox(int)::in, mailbox(int)::in,
    int::in, io::di, io::uo) is det.

ask(W, Echo, Me, N, !IO) :-
    send(W, array(1 .. N), !IO),
    receive(Me, any, after(60000), R1, !IO),
    receive(Me, any, after(60000), R2, !IO),
    receive(Me, any, after(60000), R3, !IO),
    receive(Me, any, after(60000), R4, !IO),
    send(Echo, N, !IO),
    receive(Me, any, after(60000), R5, !IO),
    io.write_int(N, !IO),
    io.write_string(":", !IO),
    list.foldl(print_answer, [R1, R2, R3, R4, R5], !IO),
    io.nl(!IO).

:- pred print_answer(received(int)::in, io::di, io::uo) is det.

print_answer(R, !IO) :-
    io.write_string(" ", !IO),
    ( if R = message(V) then io.write_int(V, !IO) else io.print(R, !IO) ).

% CHECK:1: 2 2 1 0 1
% CHECK-NEXT:2: 6 7 3 0 2
% CHECK-NEXT:10: 110 155 55 22 10
% CHECK-NEXT:1000: 1001000 1500500 500500 249750 1000
% CHECK-NEXT:100000: 10000100000 15000050000 5000050000 2499975000 100000
% CHECK-NEXT:1000000: 1000001000000 1500000500000 500000500000 249999750000 1000000

% DEVICE-LABEL: merk.proc private @process_loops.worker_4_0(
% DEVICE:         merk.foreign_call @merk_device_loop_values() : () -> !llvm.ptr
% DEVICE:         merk.foreign_call @merk_device_process_loop(%c0_i64{{.*}}, %{{.*}}, %c1_i64{{.*}}, %{{.*}}, %c0_i64{{.*}}) : (i64, i64, i64, i64, i64) -> ()
% DEVICE:         merk.foreign_call @merk_device_process_loop(%c1_i64{{.*}}, %{{.*}}, %c-1_i64{{.*}}, %{{.*}}, %c0_i64{{.*}})
% DEVICE-LABEL: merk.proc private @process_loops.add_previous_3_0(
% DEVICE:         merk.foreign_call @merk_device_process_loop(
% DEVICE-LABEL: merk.proc @merk_process_loop(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) attributes {merk.process_root
% DEVICE:         merk.switch %arg0 : i64 cases {{\[\[}}0], [1], [2], [3], [4]

% HOST: llvm.mlir.global internal constant @process_loops.merk_process_image.loops("10\00")

% Dispatch runs on its process's lane alone, and merk_process_loop on every
% lane of the worker teams (Lowering.cpp, MarkProcessCodeSingleLane).
% LLVM-DAG: define hidden void @merk_process_dispatch({{.*}}) #[[DISPATCH:[0-9]+]]
% LLVM-DAG: define hidden void @merk_process_loop({{.*}}) #[[OUT:[0-9]+]]
% LLVM-DAG: declare void @merk_device_sync_threads() #[[CONVERGENT:[0-9]+]]
% LLVM-DAG: attributes #[[CONVERGENT]] = { convergent }
% LLVM-DAG: attributes #[[DISPATCH]] = { convergent noinline "amdgpu-single-lane" }
% LLVM-DAG: attributes #[[OUT]] = { convergent noinline }
