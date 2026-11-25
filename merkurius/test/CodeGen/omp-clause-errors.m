% The clauses of %$omp parallel do that code generation does not yet
% implement, each reported for the leaf of the decomposition it applies to,
% parallel or do, and those allowed on neither (design document, "Loops",
% Under a parallel directive). Flang lowers each of them; none is silently
% dropped here.
%
% RUN: not merkc -fc1 -fopenmp -fdo-loops -emit-merk %s -o /dev/null 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module omp_clause_errors.
:- implementation.
:- import_module int.

:- pred p0(int::in, int::out) is det.
p0(K, U) :-
    %$omp parallel do private(K)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the private clause of the do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p1(int::in, int::out) is det.
p1(K, U) :-
    %$omp parallel do firstprivate(K)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the firstprivate clause of the do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p2(int::in, int::out) is det.
p2(K, U) :-
    %$omp parallel do shared(K)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the shared clause of the parallel directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p3(int::in, int::out) is det.
p3(K, U) :-
    %$omp parallel do default(none)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the default clause of the parallel directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p4(int::in, int::out) is det.
p4(K, U) :-
    %$omp parallel do if(K > 0)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the if clause of the parallel directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p5(int::in, int::out) is det.
p5(K, U) :-
    %$omp parallel do if(parallel: K > 0)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the if clause of the parallel directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p6(int::in, int::out) is det.
p6(K, U) :-
    %$omp parallel do nowait
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the nowait clause of the do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p7(int::in, int::out) is det.
p7(K, U) :-
    %$omp parallel do collapse(1)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the collapse clause of the do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p8(int::in, int::out) is det.
p8(K, U) :-
    %$omp parallel do lastprivate(K)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the lastprivate clause of the do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p9(int::in, int::out) is det.
p9(K, U) :-
    %$omp parallel do linear(K)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the linear clause of the do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p10(int::in, int::out) is det.
p10(K, U) :-
    %$omp parallel do order(concurrent)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the order clause of the do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p11(int::in, int::out) is det.
p11(K, U) :-
    %$omp parallel do allocate(K)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: the allocate clause is not allowed on the parallel do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p12(int::in, int::out) is det.
p12(K, U) :-
    %$omp parallel do num_threads(2)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the num_threads clause of the parallel directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p13(int::in, int::out) is det.
p13(K, U) :-
    %$omp parallel do proc_bind(close)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the proc_bind clause of the parallel directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p14(int::in, int::out) is det.
p14(K, U) :-
    %$omp parallel do reduction(and)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for reductions by operators other than +, *, min, max, /\, \/ and xor
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p15(int::in, int::out) is det.
p15(K, U) :-
    %$omp parallel do copyin(K)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the copyin clause of the parallel directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p16(int::in, int::out) is det.
p16(K, U) :-
    %$omp parallel do thread_limit(2)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: the thread_limit clause is not allowed on the parallel do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p17(int::in, int::out) is det.
p17(K, U) :-
    %$omp parallel do ordered
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the ordered clause of the do directive
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p18(int::in, int::out) is det.
p18(K, U) :-
    %$omp parallel do schedule(auto)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the auto schedule
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

:- pred p19(int::in, int::out) is det.
p19(K, U) :-
    %$omp parallel do schedule(runtime)
% CHECK: omp-clause-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the runtime schedule
    ( for(I, 1, K), fromto(0, U0, U1, U) do U1 = U0 + I ).

