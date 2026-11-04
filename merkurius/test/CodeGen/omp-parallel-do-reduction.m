% Loops under %$omp parallel do with reduction clauses (design document,
% "Loops", Reductions): the value a reduction names, a fromto's Last or a
% state variable, or the one scalar the loop carries, for a clause that
% names none, is a slot that holds First, or the state variable's value
% before the loop, and is the do's reduction variable, by value, under an
% omp.declare_reduction of the operator and the type, as Flang declares
% them; each iteration's body takes the do's copy and gives its next
% value. After the loop, the slot holds Last. Run on four threads, the
% program prints what the sequential program prints.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-mlir %s | FileCheck %s --check-prefix=MLIR
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module omp_parallel_do_reduction.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

main(!IO) :-
    % A float sum depends on the order it is added in, so on the schedule.
    io.write_int(float.round_to_int(pi(1000) * 1000000.0), !IO), io.nl(!IO),
    state(100, S), io.write_int(S, !IO), io.nl(!IO),
    unnamed(100, U), io.write_int(U, !IO), io.nl(!IO),
    array.init(10, 0, A0),
    both(A0, A, Lo, Hi),
    show(A, 0, !IO), io.nl(!IO),
    io.write_int(Lo, !IO), io.nl(!IO), io.write_int(Hi, !IO), io.nl(!IO),
    product(10, P), io.write_int(P, !IO), io.nl(!IO),
    bits(B), io.write_int(B, !IO), io.nl(!IO),
    largest(100, L), io.write_int(L, !IO), io.nl(!IO).
% OUTPUT:3141593
% OUTPUT:6050
% OUTPUT:10100
% OUTPUT: -3 4 1 -2 5 2 -1 6 3 0
% OUTPUT:-3
% OUTPUT:6
% OUTPUT:3628800
% OUTPUT:65535
% OUTPUT:99

% Each declaration is made once, at the module's start, in the order the
% loops need them.
% MERK-LABEL: omp.declare_reduction @add_reduction_f64 : f64 init {
% MERK-NEXT:  ^bb0(%{{.*}}: f64):
% MERK-NEXT:    %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
% MERK-NEXT:    omp.yield(%[[ZERO]] : f64)
% MERK-NEXT:  } combiner {
% MERK-NEXT:  ^bb0(%[[A:.*]]: f64, %[[B:.*]]: f64):
% MERK-NEXT:    %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : f64
% MERK-NEXT:    omp.yield(%[[SUM]] : f64)
% MERK-NEXT:  }
% MERK-NEXT:  omp.declare_reduction @add_reduction_i64 : i64 init {
% MERK:         arith.constant 0 : i64
% MERK:         arith.addi
% MERK:       omp.declare_reduction @min_i64 : i64 init {
% MERK:         arith.constant 9223372036854775807 : i64
% MERK:         arith.minsi
% MERK:       omp.declare_reduction @max_i64 : i64 init {
% MERK:         arith.constant -9223372036854775808 : i64
% MERK:         arith.maxsi
% MERK:       omp.declare_reduction @multiply_reduction_i64 : i64 init {
% MERK:         arith.constant 1 : i64
% MERK:         arith.muli
% MERK:       omp.declare_reduction @ieor_i64 : i64 init {
% MERK:         arith.constant 0 : i64
% MERK:         arith.xori
% MERK-NOT:   omp.declare_reduction
% MERK:       merk.proc @omp_parallel_do_reduction.main_2_0()

% The midpoint rule for the integral of 4 / (1 + X^2) over [0, 1].
:- func pi(int) = float.
pi(N) = Step * Sum :-
    Step = 1.0 / float(N),
    %$omp parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        X = (float(I) + 0.5) * Step,
        S1 = S0 + 4.0 / (1.0 + X * X)
    ).
% MERK-LABEL: merk.proc private @omp_parallel_do_reduction.pi_f1_0(
% MERK:         %[[SLOT:.*]] = llvm.alloca %{{.*}} x f64
% MERK:         %[[FIRST:.*]] = merk.constant 0.000000e+00 : f64
% MERK:         llvm.store %[[FIRST]], %[[SLOT]] : f64, !llvm.ptr
% MERK:         omp.parallel {
% MERK:           omp.wsloop reduction(@add_reduction_f64 %[[SLOT]] -> %[[COPY:.*]] : !llvm.ptr) {
% MERK-NEXT:        omp.loop_nest (%[[I:.*]]) : i64 = {{.*}} {
% MERK-NEXT:          %[[IN:.*]] = llvm.load %[[COPY]] : !llvm.ptr -> f64
% MERK-NEXT:          %[[OUT:.*]] = merk.call @omp_parallel_do_reduction.pi__body_{{.*}}(%[[I]], %[[IN]], %{{.*}}) : (i64, f64, f64) -> f64
% MERK-NEXT:          llvm.store %[[OUT]], %[[COPY]] : f64, !llvm.ptr
% MERK-NEXT:          omp.yield
% MERK:         } {omp.combined}
% MERK-NEXT:    %[[LAST:.*]] = llvm.load %[[SLOT]] : !llvm.ptr -> f64
% MERK-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %[[LAST]] : f64

% A state variable, named by its name; S0 is combined with the partial
% sums.
:- pred state(int::in, int::out) is det.
state(N, S) :-
    S0 = 1000,
    some [!S] (
        !:S = S0,
        %$omp parallel do reduction((+) : S)
        ( for(I, 1, N) do !:S = !.S + I ),
        S = !.S
    ).
% MERK-LABEL: merk.proc private @omp_parallel_do_reduction.state_2_0(
% MERK:         omp.wsloop reduction(@add_reduction_i64 %{{.*}} -> %{{.*}} : !llvm.ptr) {

% The one scalar a loop carries, for a clause that names none.
:- pred unnamed(int::in, int::out) is det.
unnamed(N, U) :-
    %$omp parallel do reduction(+)
    ( for(I, 1, N), fromto(0, U0, U1, U) do U1 = U0 + 2 * I ).
% MERK-LABEL: merk.proc private @omp_parallel_do_reduction.unnamed_2_0(
% MERK:         omp.wsloop reduction(@add_reduction_i64 %{{.*}} -> %{{.*}} : !llvm.ptr) {

:- pred show(array(int)::in, int::in, io::di, io::uo) is det.
show(A, I, !IO) :-
    ( if I < array.size(A) then
        io.write_string(" ", !IO), io.write_int(A ^ elem(I), !IO),
        show(A, I + 1, !IO)
    else
        true
    ).

% Two reductions and an array updated in place; the identities are the
% smallest and the largest int.
:- pred both(array(int)::array_di, array(int)::array_uo, int::out, int::out)
    is det.
both(!A, Lo, Hi) :-
    %$omp parallel do reduction(min : Lo) reduction(max : Hi)
    ( for(I, 0, 9), fromto(max_int, L0, L1, Lo), fromto(min_int, H0, H1, Hi) do
        V = (I * 7) mod 10 - 3,
        !A ^ elem(I) := V,
        L1 = int.min(L0, V),
        H1 = int.max(H0, V)
    ).
% MERK-LABEL: merk.proc private @omp_parallel_do_reduction.both_4_0(
% MERK:         omp.wsloop reduction(@min_i64 %[[LO:.*]] -> %[[L:.*]], @max_i64 %[[HI:.*]] -> %[[H:.*]] : !llvm.ptr, !llvm.ptr) {
% MERK:           %[[L0:.*]] = llvm.load %[[L]] : !llvm.ptr -> i64
% MERK-NEXT:      %[[H0:.*]] = llvm.load %[[H]] : !llvm.ptr -> i64
% MERK-NEXT:      %[[OUT:.*]]:3 = merk.call @omp_parallel_do_reduction.both__body_{{.*}}(%{{.*}}, %[[L0]], %[[H0]], %{{.*}}) : (i64, i64, i64, !merk.array<i64>) -> (i64, i64, !merk.array<i64>)
% MERK-NEXT:      llvm.store %[[OUT]]#0, %[[L]] : i64, !llvm.ptr
% MERK-NEXT:      llvm.store %[[OUT]]#1, %[[H]] : i64, !llvm.ptr

:- pred product(int::in, int::out) is det.
product(N, P) :-
    %$omp parallel do reduction((*) : P)
    ( for(I, 1, N), fromto(1, P0, P1, P) do P1 = P0 * I ).

:- pred bits(int::out) is det.
bits(B) :-
    %$omp parallel do reduction(xor)
    ( for(I, 0, 15), fromto(0, B0, B1, B) do B1 = xor(B0, 1 << I) ).

% An operator qualified by the type's module.
:- pred largest(int::in, int::out) is det.
largest(N, L) :-
    %$omp parallel do reduction(int.max)
    ( for(I, 0, N - 1), fromto(min_int, L0, L1, L) do L1 = int.max(L0, I) ).

% MLIR-LABEL: llvm.func internal tailcc @omp_parallel_do_reduction.pi_f1_0(
% MLIR:         omp.wsloop reduction(@add_reduction_f64 %{{.*}} -> %{{.*}} : !llvm.ptr) {
