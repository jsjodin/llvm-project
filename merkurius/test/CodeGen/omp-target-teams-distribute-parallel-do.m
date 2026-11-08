% A loop under %$omp target teams distribute parallel do is a target
% region holding omp.teams around omp.parallel, omp.distribute and
% omp.wsloop over one omp.loop_nest, as Flang lowers !$omp target teams
% distribute parallel do: target and teams are combined, and parallel,
% distribute and do, the parts of the composite distribute parallel do,
% are composite. The region maps the arrays and the values the loop uses,
% as any target region does. The kernel is SPMD: the host computes the
% bounds and the step before the region and gives them to it as host_eval
% values, which the nest runs over. The bounds are computed in omp.parallel
% too, where the region reads the values it is given, as Flang places
% them, so that the device, which has no host_eval, maps the same values
% and its nest runs over those.
% omp-target-teams-distribute-parallel-do-gpu.test runs the program on the
% GPU; here it runs on the host, and as the sequential program.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-mlir -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MLIR
% RUN: merkc -fc1 -fopenmp -fdo-loops --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fdo-loops -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE-MERK
% RUN: merkc -fc1 -fopenmp -fdo-loops -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_teams_distribute_parallel_do.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

main(!IO) :-
    % More iterations than a team has threads.
    N = 100000,
    array.init(N, 0.0, X0),
    iota(X0, X),
    array.init(N, 1.0, Y0),
    saxpy(3.0, X, Y0, Y),
    io.write_float(Y ^ elem(0), !IO), io.nl(!IO),
    io.write_float(Y ^ elem(N - 1), !IO), io.nl(!IO),
    sum(Y, 0, N - 1, 0.0, S),
    io.write_float(S, !IO), io.nl(!IO),
    array.init(10, 0, A0),
    down(A0, A1),
    stride(A1, A2),
    empty(A2, A),
    show(A, 0, !IO), io.nl(!IO),
    array.init(8, 0, B0),
    array.init(8, 0, C0),
    both(B0, B, C0, C),
    show(B, 0, !IO), io.nl(!IO),
    show(C, 0, !IO), io.nl(!IO).

% X ^ elem(I) = I, sequentially.
:- pred iota(array(float)::array_di, array(float)::array_uo) is det.
iota(!X) :-
    ( for(I, 0, array.size(!.X) - 1) do !X ^ elem(I) := float(I) ).

% Shared variables, A and X, after the counter and the array; X is mapped
% to, Y tofrom, A and N by copy. Y ^ elem(I) = 3 * I + 1.
:- pred saxpy(float::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
saxpy(A, X, !Y) :-
    N = array.size(X),
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !Y ^ elem(I) := A * X ^ elem(I) + !.Y ^ elem(I)
    ).
% The host computes the bounds and the step from N, and maps N too.
% MERK-LABEL: merk.proc private @omp_target_teams_distribute_parallel_do.saxpy_4_0(
% MERK:         %[[SIZE:.*]] = merk.array_size %arg1
% MERK-NEXT:    %[[HLO:.*]] = merk.constant 0 : i64
% MERK:         %[[HHI:.*]] = arith.subi %[[SIZE]], %{{.*}} : i64
% MERK-NEXT:    %[[HSTEP:.*]] = arith.constant 1 : i64
% MERK:         omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(to) capture(ByCopy) name("N")
% MERK:         omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(tofrom) capture(ByRef) bounds(%{{.*}}) name("STATE_VARIABLE_Y_0")
% MERK:         omp.map.info var_ptr(%{{.*}} : !llvm.ptr, f64) map_clauses(to) capture(ByCopy) name("A")
% MERK:         omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(to) capture(ByRef) bounds(%{{.*}}) name("X")
% MERK:         omp.target kernel_type(spmd) host_eval(%[[HLO]] -> %[[LO:.*]], %[[HHI]] -> %[[HI:.*]], %[[HSTEP]] -> %[[STEP:.*]] : i64, i64, i64) map_entries(%{{.*}} -> %[[N:.*]], %{{.*}} -> %[[Y:.*]], %{{.*}} -> %[[A:.*]], %{{.*}} -> %[[X:.*]] : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
% MERK-NEXT:      omp.teams {
% MERK-NEXT:        omp.parallel {
% MERK:               %[[NV:.*]] = llvm.load %[[N]] : !llvm.ptr -> i64
% MERK:               %[[AV:.*]] = llvm.load %[[A]] : !llvm.ptr -> f64
% MERK:               arith.subi %[[NV]], %{{.*}} : i64
% MERK:               omp.distribute {
% MERK-NEXT:            omp.wsloop {
% MERK-NEXT:              omp.loop_nest (%[[I:.*]]) : i64 = (%[[LO]]) to (%[[HI]]) inclusive step (%[[STEP]]) {
% MERK-NEXT:                merk.call @omp_target_teams_distribute_parallel_do.saxpy__body_{{[0-9]+}}_7_5_0(%[[I]], %{{.*}}, %[[AV]], %{{.*}})
% MERK-NEXT:                omp.yield
% MERK-NEXT:              }
% MERK-NEXT:            } {omp.composite}
% MERK-NEXT:          } {omp.composite}
% MERK-NEXT:          omp.terminator
% MERK-NEXT:        } {omp.composite}
% MERK-NEXT:        omp.terminator
% MERK-NEXT:      } {omp.combined}
% MERK-NEXT:      omp.terminator
% MERK-NEXT:    } {omp.combined}
% MERK-NOT:   merk.call @omp_target_teams_distribute_parallel_do.saxpy__loop
% The device has no host_eval: its nest runs over the bounds it computes.
% DEVICE-MERK-LABEL: merk.proc private @omp_target_teams_distribute_parallel_do.saxpy_4_0(
% DEVICE-MERK:         omp.target kernel_type(spmd) map_entries(%{{[0-9]+}} -> %[[N:[a-z0-9]+]], {{.*}}) {
% DEVICE-MERK:             %[[NV:.*]] = llvm.load %[[N]] : !llvm.ptr -> i64
% DEVICE-MERK:             %[[LO:.*]] = merk.constant 0 : i64
% DEVICE-MERK:             %[[HI:.*]] = arith.subi %[[NV]], %{{.*}} : i64
% DEVICE-MERK:             %[[STEP:.*]] = arith.constant 1 : i64
% DEVICE-MERK:                 omp.loop_nest (%{{.*}}) : i64 = (%[[LO]]) to (%[[HI]]) inclusive step (%[[STEP]]) {
% MLIR-LABEL: llvm.func internal tailcc @omp_target_teams_distribute_parallel_do.saxpy_4_0(
% MLIR:         omp.target kernel_type(spmd) host_eval(
% MLIR:                     omp.loop_nest
% MLIR:                  llvm.call tailcc @omp_target_teams_distribute_parallel_do.saxpy__body_{{[0-9]+}}_7_5_0(
% The body procedure is on the device, called from the loop's body.
% DEVICE-LABEL: define internal ptr @omp_target_teams_distribute_parallel_do.saxpy__body_
% DEVICE: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_omp_target_teams_distribute_parallel_do.saxpy_4_0_l{{[0-9]+}}(
% DEVICE: call void @__kmpc_parallel_60(
% DEVICE: call void @__kmpc_distribute_for_static_loop_8u(

% A negative step: the elements from 9 down to 5.
:- pred down(array(int)::array_di, array(int)::array_uo) is det.
down(!A) :-
    %$omp target teams distribute parallel do
    ( for(I, 9, 5, -1) do !A ^ elem(I) := I ).
% MERK-LABEL: merk.proc private @omp_target_teams_distribute_parallel_do.down_2_0(
% MERK-NEXT:    %[[HLO:.*]] = merk.constant 9 : i64
% MERK-NEXT:    %[[HHI:.*]] = merk.constant 5 : i64
% MERK-NEXT:    %[[HSTEP:.*]] = arith.constant -1 : i64
% MERK:         omp.target kernel_type(spmd) host_eval(%[[HLO]] -> %[[LO:.*]], %[[HHI]] -> %[[HI:.*]], %[[HSTEP]] -> %[[STEP:.*]] : i64, i64, i64)
% MERK:         omp.loop_nest (%{{.*}}) : i64 = (%[[LO]]) to (%[[HI]]) inclusive step (%[[STEP]]) {

% Every third element from 0: 0, 3, 6 and 9, which down set already.
:- pred stride(array(int)::array_di, array(int)::array_uo) is det.
stride(!A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, 9, 3) do !A ^ elem(I) := !.A ^ elem(I) + 100 ).

% No iterations.
:- pred empty(array(int)::array_di, array(int)::array_uo) is det.
empty(!A) :-
    %$omp target teams distribute parallel do
    ( for(I, 5, 4) do !A ^ elem(I) := -1 ).

% Two arrays updated in place, both mapped tofrom.
:- pred both(array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo) is det.
both(!B, !C) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, 7) do !B ^ elem(I) := I, !C ^ elem(I) := 2 * I ).
% MERK-LABEL: merk.proc private @omp_target_teams_distribute_parallel_do.both_4_0(
% MERK:         omp.map.info {{.*}} map_clauses(tofrom) capture(ByRef) {{.*}} name("STATE_VARIABLE_B_0")
% MERK:         omp.map.info {{.*}} map_clauses(tofrom) capture(ByRef) {{.*}} name("STATE_VARIABLE_C_0")
% MERK:         merk.succeed %arg0, %arg1

:- pred sum(array(float)::in, int::in, int::in, float::in, float::out) is det.
sum(A, I, Hi, S0, S) :-
    ( if I > Hi then S = S0 else sum(A, I + 1, Hi, S0 + A ^ elem(I), S) ).

:- pred show(array(int)::in, int::in, io::di, io::uo) is det.
show(A, I, !IO) :-
    ( if I < array.size(A) then
        io.write_string(" ", !IO), io.write_int(A ^ elem(I), !IO),
        show(A, I + 1, !IO)
    else
        true
    ).

% OUTPUT:1.0
% OUTPUT-NEXT:299998.0
% OUTPUT-NEXT:14999950000.0
% OUTPUT-NEXT: 100 0 0 100 0 5 106 7 8 109
% OUTPUT-NEXT: 0 1 2 3 4 5 6 7
% OUTPUT-NEXT: 0 2 4 6 8 10 12 14
