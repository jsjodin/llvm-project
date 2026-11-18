% Atomic updates of array elements in the loops that a GPU process runs
% across its teams (docs/INPROGRESS.md, Atomic updates of array elements):
% omp.atomic.update and omp.atomic.capture in the loops' bodies, which the
% device's compile, without OpenMP's runtime, translates to atomicrmw. A
% tallier answers each number N with what loops over 0 .. N - 1 made: a
% count of each digit, as ints; a sum of halves in each of four floats; the
% odd numbers appended to an array at the indices a capture takes from a
% count; and the mismatches of a count of each number appended, which a
% second loop of atomic updates makes, against one for each odd number,
% counted by a reduction. The program runs the kind at the host here, where
% the loops are parallel do loops, or sequential without -fopenmp, and at
% the GPU in process-atomics-gpu.test. On an AMD GPU the updates are on
% global pointers, which the backend combines across a wave, and the loops'
% values are read from the wave's first lane, so that an array's address is
% known to be the same in every lane.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t
% RUN: %libomp-env OMP_NUM_THREADS=4 %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t.sequential
% RUN: %libomp-env %t.sequential host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t.sequential2
% RUN: %libomp-env %t.sequential2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_atomics.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process.

:- type results
    --->    results(int, int, int, int, int, int).

    % Answers each number with what its loops made, until 0.
:- process tallier(mailbox(results)) receives int on gpu.

tallier(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N > 0 then
        array.init(10, 0, H0),
        %$omp teams distribute parallel do
        ( for(I, 0, N - 1), fromto(H0, H1, H2, H) do
            K = I mod 10,
            %$omp atomic
            H2 = H1 ^ elem(K) := H1 ^ elem(K) + 1
        ),
        array.init(4, 0.0, F0),
        %$omp teams distribute parallel do
        ( for(I, 0, N - 1), fromto(F0, F1, F2, F) do
            add_half(I mod 4, F1, F2)
        ),
        array.init(1, 0, C0),
        array.init(N / 2 + 1, -1, V0),
        %$omp teams distribute parallel do
        ( for(I, 0, N - 1), fromto(C0, C1, C2, C), fromto(V0, V1, V2, V) do
            ( if I mod 2 = 1 then
                %$omp atomic capture
                ( At = C1 ^ elem(0), C2 = C1 ^ elem(0) := At + 1 ),
                V2 = V1 ^ elem(At) := I
            else
                C2 = C1,
                V2 = V1
            )
        ),
        Count = C ^ elem(0),
        array.init(N, 0, S0),
        %$omp teams distribute parallel do
        ( for(J, 0, Count - 1), fromto(S0, S1, S2, S) do
            X = V ^ elem(J),
            %$omp atomic
            S2 = S1 ^ elem(X) := S1 ^ elem(X) + 1
        ),
        %$omp teams distribute parallel do reduction((+) : Wrong)
        ( for(I, 0, N - 1), fromto(0, W0, W1, Wrong) do
            W1 = ( if S ^ elem(I) = I mod 2 then W0 else W0 + 1 )
        ),
        send(Reply, results(H ^ elem(0), H ^ elem(9),
            truncate_to_int(F ^ elem(0) * 2.0),
            truncate_to_int(F ^ elem(3) * 2.0), Count, Wrong), !IO),
        tallier(Reply, Self, !IO)
    else
        true
    ).

:- pred add_half(int::in, array(float)::array_di, array(float)::array_uo)
    is det.

add_half(K, !F) :-
    %$omp atomic update
    !F ^ elem(K) := !.F ^ elem(K) + 0.5.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Me, !IO),
    spawn(Place, tallier(Me), Tallier, !IO),
    list.foldl(ask(Tallier, Me), [1, 2, 10, 1000, 100000, 1000000], !IO),
    send(Tallier, 0, !IO).

:- pred ask(mailbox(int)::in, mailbox(results)::in, int::in,
    io::di, io::uo) is det.

ask(Tallier, Me, N, !IO) :-
    send(Tallier, N, !IO),
    receive(Me, any, after(60000), R, !IO),
    io.write_int(N, !IO),
    io.write_string(":", !IO),
    ( if R = message(results(A, B, C, D, E, F)) then
        list.foldl(print_result, [A, B, C, D, E, F], !IO)
    else
        io.write_string(" ", !IO),
        io.print(R, !IO)
    ),
    io.nl(!IO).

:- pred print_result(int::in, io::di, io::uo) is det.

print_result(V, !IO) :-
    io.write_string(" ", !IO),
    io.write_int(V, !IO).

% CHECK:1: 1 0 1 0 0 0
% CHECK-NEXT:2: 1 0 1 0 1 0
% CHECK-NEXT:10: 1 1 3 2 5 0
% CHECK-NEXT:1000: 100 100 250 250 500 0
% CHECK-NEXT:100000: 10000 10000 25000 25000 50000 0
% CHECK-NEXT:1000000: 100000 100000 250000 250000 500000 0

% DEVICE-LABEL: merk.proc private @process_atomics.add_half_3_0(
% DEVICE:         omp.atomic.update %{{.*}} : !llvm.ptr<1> {
% DEVICE:         arith.addf
% LLVM-LABEL: define {{.*}} @process_atomics.add_half_3_0(
% LLVM:         atomicrmw fadd ptr addrspace(1) %{{.*}}, double 5.000000e-01 monotonic
% LLVM-LABEL: define {{.*}} @merk_process_loop(
% LLVM:         call i64 @llvm.amdgcn.readfirstlane.i64(
