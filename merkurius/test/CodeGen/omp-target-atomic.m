% Atomic updates of array elements in loops under %$omp target teams
% distribute parallel do (docs/INPROGRESS.md, Atomic updates of array
% elements): the device's compile has the same omp.atomic.update and
% omp.atomic.capture as the host's, which its translation makes atomicrmw
% on the GPU. A count of each digit, a sum of halves in a predicate the loop
% calls, and the odd numbers appended at the indices a capture takes. Run on
% the host's threads, the program prints what the sequential program
% prints; omp-target-atomic-gpu.test runs it on the GPU.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fdo-loops -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE-MERK
% RUN: merkc -fc1 -fopenmp -fdo-loops -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env OMP_NUM_THREADS=4 %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module omp_target_atomic.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

main(!IO) :-
    N = 1000000,
    array.init(10, 0, H0),
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1), fromto(H0, H1, H2, H) do
        K = I mod 10,
        %$omp atomic
        H2 = H1 ^ elem(K) := H1 ^ elem(K) + 1
    ),
    array.init(4, 0.0, F0),
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1), fromto(F0, F1, F2, F) do
        add_half(I mod 4, F1, F2)
    ),
    array.init(1, 0, C0),
    array.init(N / 2, -1, V0),
    %$omp target teams distribute parallel do
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
    io.write_int(H ^ elem(0), !IO), io.write_string(" ", !IO),
    io.write_int(H ^ elem(9), !IO), io.nl(!IO),
    io.write_float(F ^ elem(0), !IO), io.write_string(" ", !IO),
    io.write_float(F ^ elem(3), !IO), io.nl(!IO),
    io.write_int(C ^ elem(0), !IO), io.write_string(" ", !IO),
    io.write_int(sum_odds(V, 0, 0), !IO), io.nl(!IO).
% OUTPUT:100000 100000
% OUTPUT-NEXT:125000.0 125000.0
% OUTPUT-NEXT:500000 250000000000

:- pred add_half(int::in, array(float)::array_di, array(float)::array_uo)
    is det.

add_half(K, !F) :-
    %$omp atomic update
    !F ^ elem(K) := !.F ^ elem(K) + 0.5.

    % The sum of the odd elements, the odd numbers below N if each was
    % appended once.
:- func sum_odds(array(int), int, int) = int.

sum_odds(V, J, S) =
    ( if J < array.size(V) then
        sum_odds(V, J + 1, ( if V ^ elem(J) mod 2 = 1 then S + V ^ elem(J) else S ))
    else
        S
    ).

% The loops' bodies, which the kernels call.
% DEVICE-MERK:       merk.proc private @omp_target_atomic.main__body_
% DEVICE-MERK:         omp.atomic.update %{{.*}} : !llvm.ptr {
% DEVICE-MERK:       merk.proc private @omp_target_atomic.main__body_
% DEVICE-MERK:       merk.proc private @omp_target_atomic.main__body_
% DEVICE-MERK:         omp.atomic.capture {
% DEVICE-MERK-LABEL: merk.proc {{.*}}@omp_target_atomic.add_half_3_0(
% DEVICE-MERK:         omp.atomic.update %{{.*}} : !llvm.ptr {
%
% DEVICE: atomicrmw add ptr %{{.*}}, i64 1 monotonic
% DEVICE: atomicrmw add ptr %{{.*}}, i64 1 monotonic
% DEVICE-LABEL: define {{.*}} @omp_target_atomic.add_half_3_0(
% DEVICE: atomicrmw fadd ptr %{{.*}}, double 5.000000e-01 monotonic
