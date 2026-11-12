% Reductions in the loops that a GPU process runs across its teams, under
% distribute parallel do (docs/INPROGRESS.md, GPU processes that run loops):
% each thread keeps its partial results, a team combines its threads', and
% the process combines the teams' with the values before the loop. A reducer
% answers each number N with the results of loops over 1 .. N: a sum from a
% value other than the identity; a minimum and a maximum in one loop; a sum
% of floats; an exclusive or; a count beside an array updated in place; a
% product of at most 20 numbers; a maximum of floats; and a sum of a loop
% with no iterations, which is the value before it. The program runs the
% kind at the host here, where the loops are parallel do loops, or
% sequential without -fopenmp, and at the GPU in
% process-loop-reductions-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% The device's compile stores the values before a loop in the words of its
% results, and passes merk_device_process_loop the number of reductions;
% merk_process_loop runs a thread's rounds of a loop, in a procedure of
% their own that carries its partial results, ending each round with a
% barrier across the team; merk_process_loop_start sets a thread's partial
% results to the operators' identities, and merk_process_loop_reduce
% combines them.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t.sequential
% RUN: %libomp-env %t.sequential host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t.sequential2
% RUN: %libomp-env %t.sequential2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_loop_reductions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process.

:- type results
    --->    results(int, int, int, int, int, int, int, int, int, int).

    % Answers each number with the results of its loops, until 0.
:- process reducer(mailbox(results)) receives int on gpu.

reducer(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N > 0 then
        %$omp distribute parallel do reduction((+) : Sum)
        ( for(I, 1, N), fromto(10, S0, S1, Sum) do
            S1 = S0 + I
        ),
        %$omp distribute parallel do reduction(min : Lo) reduction(max : Hi)
        ( for(I, 1, N), fromto(max_int, Lo0, Lo1, Lo),
            fromto(min_int, Hi0, Hi1, Hi) do
            V = (I * 7919) mod 100003,
            Lo1 = int.min(Lo0, V),
            Hi1 = int.max(Hi0, V)
        ),
        %$omp distribute parallel do reduction((+) : F)
        ( for(I, 1, N), fromto(0.0, F0, F1, F) do
            F1 = F0 + float(I) * 0.5
        ),
        %$omp distribute parallel do reduction(xor : X)
        ( for(I, 1, N), fromto(0, X0, X1, X) do
            X1 = xor(X0, I * 3)
        ),
        array.init(N, 0, A0),
        %$omp distribute parallel do reduction((+) : Evens)
        ( for(I, 0, N - 1), fromto(A0, A1, A2, A), fromto(0, E0, E1, Evens) do
            array.set(I, I * I, A1, A2),
            E1 = ( if I mod 2 = 0 then E0 + 1 else E0 )
        ),
        %$omp distribute parallel do reduction((*) : P)
        ( for(I, 1, int.min(N, 20)), fromto(1, P0, P1, P) do
            P1 = P0 * I
        ),
        %$omp distribute parallel do reduction(max : G)
        ( for(I, 1, N), fromto(-1.0, G0, G1, G) do
            G1 = float.max(G0, float(I mod 17) * 1.5)
        ),
        %$omp distribute parallel do reduction((+) : Z)
        ( for(I, 1, 0), fromto(42, Z0, Z1, Z) do
            Z1 = Z0 + I
        ),
        send(Reply, results(Sum, Lo, Hi, truncate_to_int(F), X, Evens,
            array.lookup(A, N - 1), P, truncate_to_int(G * 10.0), Z), !IO),
        reducer(Reply, Self, !IO)
    else
        true
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Me, !IO),
    spawn(Place, reducer(Me), Reducer, !IO),
    list.foldl(ask(Reducer, Me), [1, 2, 10, 1000, 100000, 1000000], !IO),
    send(Reducer, 0, !IO).

:- pred ask(mailbox(int)::in, mailbox(results)::in, int::in,
    io::di, io::uo) is det.

ask(Reducer, Me, N, !IO) :-
    send(Reducer, N, !IO),
    receive(Me, any, after(60000), R, !IO),
    io.write_int(N, !IO),
    io.write_string(":", !IO),
    ( if R = message(results(A, B, C, D, E, F, G, H, I, J)) then
        list.foldl(print_result, [A, B, C, D, E, F, G, H, I, J], !IO)
    else
        io.write_string(" ", !IO),
        io.print(R, !IO)
    ),
    io.nl(!IO).

:- pred print_result(int::in, io::di, io::uo) is det.

print_result(V, !IO) :-
    io.write_string(" ", !IO),
    io.write_int(V, !IO).

% CHECK:1: 11 7919 7919 0 3 1 0 1 15 42
% CHECK-NEXT:2: 13 7919 15838 1 5 1 1 2 30 42
% CHECK-NEXT:10: 65 7919 79190 27 21 5 81 3628800 150 42
% CHECK-NEXT:1000: 500510 93 99984 250250 2000 500 998001 2432902008176640000 240 42
% CHECK-NEXT:100000: 5000050010 1 100002 2500025000 466240 50000 9999800001 2432902008176640000 240 42
% CHECK-NEXT:1000000: 500000500010 0 100002 250000250000 1836032 500000 999998000001 2432902008176640000 240 42

% DEVICE-LABEL: merk.proc private @process_loop_reductions.reducer_4_0(
% DEVICE:         %[[RESULTS:.*]] = merk.foreign_call @merk_device_loop_results() : () -> !llvm.ptr
% DEVICE:         llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr
% DEVICE:         merk.foreign_call @merk_device_process_loop(%c0_i64{{.*}}, %{{.*}}, %c1_i64{{.*}}, %{{.*}}, %c1_i64{{.*}}) : (i64, i64, i64, i64, i64) -> ()
% DEVICE:         merk.foreign_call @merk_device_loop_results() : () -> !llvm.ptr
% DEVICE:         merk.foreign_call @merk_device_process_loop(%c1_i64{{.*}}, %{{.*}}, %c1_i64{{.*}}, %{{.*}}, %c2_i64{{.*}})
% DEVICE-LABEL: merk.proc @merk_process_loop(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) attributes {merk.process_root
% DEVICE:         merk.call @[[SUM:process_loop_reductions.reducer__body_[0-9]+_11_3_0]]$rounds(%arg2, %arg1, %arg3, %arg4, %arg5, %arg6, %{{.*}}) : (i64, i64, i64, i64, i64, i64, i64) -> i64
% DEVICE:         merk.call @[[MINMAX:process_loop_reductions.reducer__body_[0-9]+_11_5_0]]$rounds(%arg2, %arg1, %arg3, %arg4, %arg5, %arg6, %{{.*}}, %{{.*}}) : (i64, i64, i64, i64, i64, i64, i64, i64) -> (i64, i64)
% DEVICE-LABEL: merk.proc private @process_loop_reductions.reducer__body_{{[0-9]+}}_11_5_0$rounds(
% DEVICE:         merk.ite -> i64, i64 {
% DEVICE:           arith.cmpi sgt, %arg0, %c0_i64
% DEVICE:         } then {
% DEVICE:           %[[NEXT:.*]]:2 = merk.ite -> i64, i64 {
% DEVICE:             arith.cmpi slt, %arg1, %arg3
% DEVICE:           } then {
% DEVICE:             %[[TIMES:.*]] = arith.muli %arg1, %arg5
% DEVICE:             %[[COUNTER:.*]] = arith.addi %arg4, %[[TIMES]]
% DEVICE:             merk.call @[[MINMAX]](%[[COUNTER]], %arg6, %arg7)
% DEVICE:           } else {
% DEVICE:             merk.yield %arg6, %arg7
% DEVICE:           }
% DEVICE:           merk.foreign_call @merk_device_sync_threads()
% DEVICE:           merk.call @[[MINMAX]]$rounds(%{{.*}}, %{{.*}}, %arg2, %arg3, %arg4, %arg5, %[[NEXT]]#0, %[[NEXT]]#1)
% DEVICE:         } else {
% DEVICE:           merk.yield %arg6, %arg7
% DEVICE-LABEL: merk.proc @merk_process_loop_start(%arg0: i64, %arg1: i64) attributes {merk.process_root
% DEVICE:         arith.constant 0 : i64
% DEVICE:       }, {
% DEVICE:         arith.constant 9223372036854775807 : i64
% DEVICE:         arith.constant -9223372036854775808 : i64
% DEVICE:       }, {
% DEVICE:         arith.constant 0.000000e+00 : f64
% DEVICE-LABEL: merk.proc @merk_process_loop_reduce(%arg0: i64, %arg1: i64, %arg2: i64) attributes {merk.process_root
% DEVICE:         arith.addi
% DEVICE:       }, {
% DEVICE:         arith.minsi
% DEVICE:         arith.maxsi
% DEVICE:       }, {
% DEVICE:         arith.addf
% DEVICE:       }, {
% DEVICE:         arith.xori
