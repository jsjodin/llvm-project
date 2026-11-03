% A loop under %$omp parallel do (design document, "Loops", Under a
% parallel directive) is omp.parallel around omp.wsloop over an
% omp.loop_nest, as Flang lowers !$omp parallel do. The nest runs the for's
% counter from Lo to Hi, inclusive, by the step, and each iteration calls
% the loop's body procedure with the counter, the threaded array as it was
% before the loop and the shared variables; the loop procedure is not
% called. After the loop the array is the one given, which the iterations
% updated in place. A negative step is the nest's, with Lo above Hi. The
% directive is decomposed into its leaves, as Flang decomposes it, and
% omp.parallel, which is not the innermost, is combined.
% omp-parallel-do-run.test runs the program with libomp.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk %s -o %t.mlir
% RUN: FileCheck %s --check-prefix=MERK < %t.mlir
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-mlir %s | FileCheck %s --check-prefix=MLIR
% The merk dump lowers as the module would have.
% RUN: merkc -fc1 -emit-mlir %t.mlir | FileCheck %s --check-prefix=MLIR

:- module omp_parallel_do.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

main(!IO) :-
    N = 1000,
    array.init(N, 1.0, Y0),
    array.init(N, 2.0, X),
    saxpy(3.0, X, Y0, Y),
    io.write_float(Y ^ elem(0), !IO), io.nl(!IO),
    io.write_float(Y ^ elem(N - 1), !IO), io.nl(!IO),
    sum(Y, 0, N - 1, 0.0, S),
    io.write_float(S, !IO), io.nl(!IO),
    array.init(10, 0, A0),
    down(A0, A1),
    stride(A1, A2),
    empty(A2, A),
    show(A, 0, !IO), io.nl(!IO).

% Shared variables, A and X, after the counter and the array.
:- pred saxpy(float::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
saxpy(A, X, !Y) :-
    N = array.size(X),
    %$omp parallel do
    ( for(I, 0, N - 1) do
        !Y ^ unsafe_elem(I) := A * X ^ unsafe_elem(I) + !.Y ^ unsafe_elem(I)
    ).
% MERK-LABEL: merk.proc private @omp_parallel_do.saxpy_4_0(%arg0: f64, %arg1: !merk.array<f64>, %arg2: !merk.array<f64>) -> !merk.array<f64>
% MERK:         %[[SIZE:.*]] = merk.array_size %arg1
% MERK:         %[[LO:.*]] = merk.constant 0 : i64
% MERK:         %[[HI:.*]] = arith.subi %[[SIZE]], %{{.*}} : i64
% MERK:         %[[STEP:.*]] = arith.constant 1 : i64
% MERK-NEXT:    omp.parallel {
% MERK-NEXT:      omp.wsloop {
% MERK-NEXT:        omp.loop_nest (%[[I:.*]]) : i64 = (%[[LO]]) to (%[[HI]]) inclusive step (%[[STEP]]) {
% MERK-NEXT:          merk.call @omp_parallel_do.saxpy__body_{{[0-9]+}}_7_5_0(%[[I]], %arg2, %arg0, %arg1)
% MERK-NEXT:          omp.yield
% MERK-NEXT:        }
% MERK-NEXT:      }
% MERK-NEXT:      omp.terminator
% MERK-NEXT:    } {omp.combined}
% MERK-NEXT:    merk.succeed %arg2 : !merk.array<f64>
% MERK-NOT:   merk.call @omp_parallel_do.saxpy__loop

% A negative step: the elements from 9 down to 5.
:- pred down(array(int)::array_di, array(int)::array_uo) is det.
down(!A) :-
    %$omp parallel do
    ( for(I, 9, 5, -1) do !A ^ unsafe_elem(I) := I ).
% MERK-LABEL: merk.proc private @omp_parallel_do.down_2_0(
% MERK:         %[[LO:.*]] = merk.constant 9 : i64
% MERK-NEXT:    %[[HI:.*]] = merk.constant 5 : i64
% MERK-NEXT:    %[[STEP:.*]] = arith.constant -1 : i64
% MERK:         omp.loop_nest (%{{.*}}) : i64 = (%[[LO]]) to (%[[HI]]) inclusive step (%[[STEP]]) {
% MLIR-LABEL: llvm.func internal tailcc @omp_parallel_do.down_2_0(
% MLIR:         omp.parallel {
% MLIR-NEXT:      omp.wsloop {
% MLIR-NEXT:        omp.loop_nest (%[[I:.*]]) : i64 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
% MLIR-NEXT:          llvm.call tailcc @omp_parallel_do.down__body_{{[0-9]+}}_7_3_0(%[[I]], %arg0)
% MLIR-NEXT:          omp.yield
% MLIR-NEXT:        }
% MLIR-NEXT:      }
% MLIR-NEXT:      omp.terminator
% MLIR-NEXT:    } {omp.combined}
% MLIR-NEXT:    llvm.return %arg0 : !llvm.ptr

% Every third element from 0: 0, 3, 6 and 9, which down set already.
:- pred stride(array(int)::array_di, array(int)::array_uo) is det.
stride(!A) :-
    %$omp parallel do
    ( for(I, 0, 9, 3) do !A ^ unsafe_elem(I) := !.A ^ unsafe_elem(I) + 100 ).
% MERK-LABEL: merk.proc private @omp_parallel_do.stride_2_0(
% MERK:         %[[STEP:.*]] = arith.constant 3 : i64
% MERK:         omp.loop_nest (%{{.*}}) : i64 = (%{{.*}}) to (%{{.*}}) inclusive step (%[[STEP]]) {

% No iterations.
:- pred empty(array(int)::array_di, array(int)::array_uo) is det.
empty(!A) :-
    %$omp parallel do
    ( for(I, 5, 4) do !A ^ unsafe_elem(I) := -1 ).

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

% OUTPUT:7.0
% OUTPUT-NEXT:7.0
% OUTPUT-NEXT:7000.0
% OUTPUT-NEXT: 100 0 0 100 0 5 106 7 8 109
