% The schedule clause of loops under %$omp parallel do and target teams
% distribute parallel do: omp.wsloop takes static or dynamic, with a chunk
% size that is a positive integer or, outside a target region, a variable
% bound before the directive, as Flang's ClauseProcessor gives them, and
% OpenMP's runtime deals the iterations out by it. Loops of uneven work, I
% mod 97 steps each, and plain sums, each with a count, so an iteration run
% twice or not at all shows. Run on the host's threads, the program prints
% what the sequential program prints; omp-schedule-gpu.test runs its target
% regions on the GPU.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk %s -o - | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp -fdo-loops --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fdo-loops -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE-MERK
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env OMP_NUM_THREADS=4 %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module omp_schedule.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    list.foldl(run, [1, 10, 1000, 100000], !IO).

:- pred run(int::in, io::di, io::uo) is det.

run(N, !IO) :-
    C = N / 7 + 1,
    uneven_static(N, A1, A2),
    uneven_dynamic(N, B1, B2),
    chunked(N, C, D1, D2),
    target_dynamic(N, E1, E2),
    target_chunked(N, F1, F2),
    io.write_int(N, !IO),
    io.write_string(":", !IO),
    list.foldl(print_result, [A1, A2, B1, B2, D1, D2, E1, E2, F1, F2], !IO),
    io.nl(!IO).

:- pred work(int::in, int::in, int::out) is det.

work(I, !W) :-
    ( for(K, 0, I mod 97), fromto(!.W, V0, V1, !:W) do
        V1 = V0 + (I * K) mod 13
    ).

:- pred uneven_static(int::in, int::out, int::out) is det.
% MERK-LABEL: merk.proc private @omp_schedule.uneven_static_3_0(
% MERK:         omp.wsloop schedule(static) reduction(
uneven_static(N, Work, Count) :-
    %$omp parallel do schedule(static) reduction((+) : Work)
    %$omp& reduction((+) : Count)
    ( for(I, 0, N - 1), fromto(0, W0, W1, Work), fromto(0, K0, K1, Count) do
        work(I, W0, W1),
        K1 = K0 + 1
    ).

:- pred uneven_dynamic(int::in, int::out, int::out) is det.
% MERK-LABEL: merk.proc private @omp_schedule.uneven_dynamic_3_0(
% MERK:         omp.wsloop schedule(dynamic) reduction(
uneven_dynamic(N, Work, Count) :-
    %$omp parallel do schedule(dynamic) reduction((+) : Work)
    %$omp& reduction((+) : Count)
    ( for(I, 0, N - 1), fromto(0, W0, W1, Work), fromto(0, K0, K1, Count) do
        work(I, W0, W1),
        K1 = K0 + 1
    ).

    % A chunk size that the caller computed.
:- pred chunked(int::in, int::in, int::out, int::out) is det.
% MERK-LABEL: merk.proc private @omp_schedule.chunked_4_0(%arg0: i64, %arg1: i64)
% MERK:         omp.wsloop schedule(dynamic = %arg1 : i64) reduction(
chunked(N, C, Sum, Count) :-
    %$omp parallel do schedule(dynamic, C) reduction((+) : Sum)
    %$omp& reduction((+) : Count)
    ( for(I, 0, N - 1), fromto(0, S0, S1, Sum), fromto(0, K0, K1, Count) do
        S1 = S0 + I,
        K1 = K0 + 1
    ).

:- pred target_dynamic(int::in, int::out, int::out) is det.
% DEVICE-MERK-LABEL: merk.proc {{.*}}@omp_schedule.target_dynamic_3_0(
% DEVICE-MERK:         omp.wsloop schedule(dynamic) reduction(
target_dynamic(N, Work, Count) :-
    %$omp target teams distribute parallel do schedule(dynamic)
    %$omp& reduction((+) : Work) reduction((+) : Count)
    ( for(I, 0, N - 1), fromto(0, W0, W1, Work), fromto(0, K0, K1, Count) do
        work(I, W0, W1),
        K1 = K0 + 1
    ).

:- pred target_chunked(int::in, int::out, int::out) is det.
% DEVICE-MERK-LABEL: merk.proc {{.*}}@omp_schedule.target_chunked_3_0(
% DEVICE-MERK:         %[[CHUNK:.*]] = arith.constant 4 : i64
% DEVICE-MERK:         omp.wsloop schedule(dynamic = %[[CHUNK]] : i64) reduction(
target_chunked(N, Sum, Count) :-
    %$omp target teams distribute parallel do schedule(dynamic, 4)
    %$omp& reduction((+) : Sum) reduction((+) : Count)
    ( for(I, 0, N - 1), fromto(0, S0, S1, Sum), fromto(0, K0, K1, Count) do
        S1 = S0 + I * 3,
        K1 = K0 + 1
    ).

:- pred print_result(int::in, io::di, io::uo) is det.

print_result(V, !IO) :-
    io.write_string(" ", !IO),
    io.write_int(V, !IO).

% OUTPUT:1: 0 1 0 1 0 1 0 1 0 1
% OUTPUT-NEXT:10: 284 10 284 10 45 10 284 10 135 10
% OUTPUT-NEXT:1000: 262895 1000 262895 1000 499500 1000 262895 1000 1498500 1000
% OUTPUT-NEXT:100000: 26849508 100000 26849508 100000 4999950000 100000 26849508 100000 14999850000 100000
