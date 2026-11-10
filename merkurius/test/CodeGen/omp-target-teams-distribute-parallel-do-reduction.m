% Reductions of loops under %$omp target teams distribute parallel do
% (design document, "Loops", Reductions): the decomposition puts the
% clause on teams and on the do, and implies a map of its variable on
% target, which is its slot, mapped tofrom by reference, as Flang maps it.
% The target region's argument for the slot is the teams' reduction
% variable, whose copy is the do's. On the device, the reductions are the
% same. Run on the host's threads, and on the GPU (the -gpu test), the
% program prints what the sequential program prints.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp -fdo-loops --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fdo-loops -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE-MERK
% RUN: merkc -fc1 -fopenmp -fdo-loops -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env OMP_NUM_THREADS=4 %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module omp_target_teams_distribute_parallel_do_reduction.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

main(!IO) :-
    % A float sum depends on the order it is added in, so on the schedule.
    pi(1000000, P),
    io.write_int(float.round_to_int(P * 1000000.0), !IO), io.nl(!IO),
    % Sums of ones are exact in any order.
    array.init(1024, 1.0, A0),
    normalize(A0, A),
    io.write_float(A ^ elem(0), !IO), io.nl(!IO),
    io.write_float(A ^ elem(1023), !IO), io.nl(!IO),
    range(100000, Lo, Hi), io.write_int(Lo, !IO), io.nl(!IO),
    io.write_int(Hi, !IO), io.nl(!IO).
% OUTPUT:3141593
% OUTPUT:0.0009765625
% OUTPUT:0.0009765625
% OUTPUT:-49999
% OUTPUT:50000

:- pred pi(int::in, float::out) is det.
pi(N, Pi) :-
    Step = 1.0 / float(N),
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        X = (float(I) + 0.5) * Step,
        S1 = S0 + 4.0 / (1.0 + X * X)
    ),
    Pi = Step * Sum.
% MERK-LABEL: merk.proc private @omp_target_teams_distribute_parallel_do_reduction.pi_2_0(
% MERK:         %[[MAP:.*]] = omp.map.info var_ptr(%[[SLOT:[0-9]+]] : !llvm.ptr, f64) map_clauses(tofrom) capture(ByRef) name("Sum")
% MERK:         merk.foreign_call @merk_region_enter() : () -> ()
% MERK-NEXT:    omp.target kernel_type(spmd) host_eval({{.*}}) map_entries({{.*}}%[[MAP]] -> %[[ARG:[a-z0-9]+]] : {{.*}}) {
% MERK-NEXT:      omp.teams reduction(@add_reduction_f64 %[[ARG]] -> %[[TEAMS:.*]] : !llvm.ptr) {
% MERK-NEXT:        omp.parallel {
% MERK:               omp.distribute {
% MERK-NEXT:            omp.wsloop reduction(@add_reduction_f64 %[[TEAMS]] -> %[[COPY:.*]] : !llvm.ptr) {
% MERK-NEXT:              omp.loop_nest
% MERK-NEXT:                %[[IN:.*]] = llvm.load %[[COPY]] : !llvm.ptr -> f64
% MERK-NEXT:                %[[OUT:.*]] = merk.call @{{.*}}pi__body{{.*}}(%{{.*}}, %[[IN]], %{{.*}})
% MERK-NEXT:                llvm.store %[[OUT]], %[[COPY]] : f64, !llvm.ptr
% MERK:           } {omp.combined}
% MERK-NEXT:      omp.terminator
% MERK-NEXT:    } {omp.combined}
% MERK-NEXT:    merk.foreign_call @merk_region_exit() : () -> ()
% MERK-NEXT:    %[[LAST:.*]] = llvm.load %[[SLOT]] : !llvm.ptr -> f64
% MERK-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %[[LAST]] : f64
% DEVICE-MERK-LABEL: omp.declare_reduction @add_reduction_f64 : f64 init {
% DEVICE-MERK-LABEL: merk.proc private @omp_target_teams_distribute_parallel_do_reduction.pi_2_0(
% DEVICE-MERK:         omp.target kernel_type(spmd) map_entries(
% DEVICE-MERK-NEXT:      omp.teams reduction(@add_reduction_f64 %{{.*}} -> %{{.*}} : !llvm.ptr) {
% DEVICE-MERK:             omp.wsloop reduction(@add_reduction_f64 %{{.*}} -> %{{.*}} : !llvm.ptr) {
% The device combines the threads' partial sums, then the teams', through
% the runtime, as Flang's does.
% DEVICE-DAG: call {{.*}} @__kmpc_nvptx_parallel_reduce_nowait_v2(
% DEVICE-DAG: call {{.*}} @__kmpc_gpu_xteam_reduce_nowait(

% The example of the design document: the second loop reads the first's
% reduction.
:- pred normalize(array(float)::array_di, array(float)::array_uo) is det.
normalize(!A) :-
    N = array.size(!.A),
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        S1 = S0 + !.A ^ elem(I)
    ),
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ elem(I) := !.A ^ elem(I) / Sum
    ).

% Two reductions of ints, a min and a max.
:- pred range(int::in, int::out, int::out) is det.
range(N, Lo, Hi) :-
    %$omp target teams distribute parallel do reduction(min : Lo)
    %$omp& reduction(max : Hi)
    ( for(I, 0, N - 1), fromto(max_int, L0, L1, Lo), fromto(min_int, H0, H1, Hi) do
        V = I - N / 2 + 1,
        L1 = int.min(L0, V),
        H1 = int.max(H0, V)
    ).
% MERK-LABEL: merk.proc private @omp_target_teams_distribute_parallel_do_reduction.range_3_0(
% MERK:         omp.teams reduction(@min_i64 %{{.*}} -> %{{.*}}, @max_i64 %{{.*}} -> %{{.*}} : !llvm.ptr, !llvm.ptr) {
% MERK:           omp.wsloop reduction(@min_i64 %{{.*}} -> %{{.*}}, @max_i64 %{{.*}} -> %{{.*}} : !llvm.ptr, !llvm.ptr) {
