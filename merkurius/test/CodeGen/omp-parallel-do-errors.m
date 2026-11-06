% The checks of a loop under %$omp parallel do (design document, "Loops",
% Under a parallel directive): it is counted by one for; the values it
% carries from one iteration to the next are arrays updated in place or
% named in reduction clauses, by operators of int or float, each once; it
% threads no io state; and its body is det and pure. The directive is decomposed into parallel and
% do, and each clause is reported for the leaf it applies to, or as not
% allowed on any. Each error leaves its procedure out. The checks are the
% same under target teams distribute parallel do, whose leaves are target,
% teams, distribute, parallel and do.
%
% RUN: not merkc -fc1 -fopenmp -fdo-loops -emit-merk %s -o /dev/null 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module omp_parallel_do_errors.
:- interface.
:- import_module array, io.
:- pred show(int::in, io::di, io::uo) is det.
:- pred scalar(int::in, int::out) is det.
:- pred state(int::in, int::in, int::out) is det.
:- pred listed(int::in) is semidet.
:- pred counted(int::in, int::out) is det.
:- pred no_for(int::in) is semidet.
:- pred no_loop(int::in, int::out) is det.
:- pred fails(int::in) is semidet.
:- pred impure_body(int::in) is det.
:- pred schedule(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred threads(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred if_parallel(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred mapped(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred indexed(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred offloaded_io(int::in, io::di, io::uo) is det.
:- pred offloaded_no_loop(int::in, int::out) is det.
:- pred teams(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred dist_schedule(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred target_map(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred nested(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred modifier(int::in, int::out) is det.
:- pred operator(int::in, int::out) is det.
:- pred unknown(int::in, int::out) is det.
:- pred several(int::in, int::out, int::out) is det.
:- pred none(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred twice(int::in, int::out) is det.
:- pred unsigned(int::in, uint::out) is det.
:- pred bitwise(int::in, float::out) is det.
:- pred qualified(int::in, float::out) is det.
:- implementation.
:- import_module float, int, list, uint.

show(N, !IO) :-
    %$omp parallel do
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: a parallel do loop must not thread the io state, since its iterations run in any order
    ( for(I, 1, N) do io.write_int(I, !IO) ).

scalar(N, S) :-
    %$omp parallel do
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:21: error: a parallel do loop can carry a fromto's value from one iteration to the next only if it is named in a reduction clause or is an array updated in place

state(N, !S) :-
    %$omp parallel do
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: a parallel do loop can carry a state variable from one iteration to the next only if it is named in a reduction clause or is an array updated in place
    ( for(I, 1, N) do !:S = !.S + I ).

% In lockstep with a list, a loop is semidet (Semantics/loops/determinism.m).
listed(N) :-
    %$omp parallel do
    ( for(I, 0, 2), foreach(X, [1, 2, 3]) do X > N + I ).
% The loop's predicates are generated too, as any others are.
% CHECK: omp-parallel-do-errors.m:[[@LINE-2]]:21: error: a parallel do loop is counted by its for, so it cannot have foreach

counted(N, M) :-
    %$omp parallel do
    ( for(_, 1, N), count(_, 1, M) do true ).
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:21: error: a parallel do loop is counted by its for, so it cannot have count

no_for(N) :-
    %$omp parallel do
    ( fromto(N, I0, I1, 0) do I0 > 0, I1 = I0 - 1 ).
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:7: error: a parallel do loop is counted by its for, so it cannot have a fromto that ends the loop

no_loop(N, M) :-
    %$omp parallel do
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: the parallel do directive must be before a loop
    M = N + 1.

fails(N) :-
    %$omp parallel do
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: the body of a parallel do loop must be det, but it is semidet
    ( for(I, 1, N) do I > 0 ).

impure_body(N) :-
    promise_pure (
        %$omp parallel do
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:9: error: the body of a parallel do loop must be pure
        ( for(I, 1, N) do impure poke(I) )
    ).

:- impure pred poke(int::in) is det.
:- pragma foreign_proc("C", poke(I::in), [will_not_call_mercury],
    "(void) I;").

schedule(N, !A) :-
    %$omp parallel do schedule(static)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the schedule clause of the do directive
    ( for(I, 0, N) do !A ^ elem(I) := I ).

threads(N, !A) :-
    %$omp parallel do num_threads(2)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the num_threads clause of the parallel directive
    ( for(I, 0, N) do !A ^ elem(I) := I ).

% A modifier names the leaf.
if_parallel(N, !A) :-
    %$omp parallel do if(parallel: N > 3)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the if clause of the parallel directive
    ( for(I, 0, N) do !A ^ elem(I) := I ).

mapped(N, !A) :-
    %$omp parallel do map(to: N)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: the map clause is not allowed on the parallel do directive
    ( for(I, 0, N) do !A ^ elem(I) := I ).

% Merkurius's own clauses take no part in the decomposition.
indexed(N, !A) :-
    %$omp parallel do index(I)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the index clause
    ( for(I, 0, N) do !A ^ elem(I) := I ).

offloaded_io(N, !IO) :-
    %$omp target teams distribute parallel do
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: a parallel do loop must not thread the io state, since its iterations run in any order
    ( for(I, 1, N) do io.write_int(I, !IO) ).

offloaded_no_loop(N, M) :-
    %$omp target teams distribute parallel do
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: the target teams distribute parallel do directive must be before a loop
    M = N + 1.

teams(N, !A) :-
    %$omp target teams distribute parallel do num_teams(4)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the num_teams clause of the teams directive
    ( for(I, 0, N) do !A ^ elem(I) := I ).

dist_schedule(N, !A) :-
    %$omp target teams distribute parallel do dist_schedule(static)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the dist_schedule clause of the distribute directive
    ( for(I, 0, N) do !A ^ elem(I) := I ).

target_map(N, !A) :-
    %$omp target teams distribute parallel do map(to: N)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the map clause of the target directive
    ( for(I, 0, N) do !A ^ elem(I) := I ).

nested(N, !A) :-
    %$omp target
    (
        %$omp target teams distribute parallel do
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:9: error: not yet implemented: code generation for target regions in target regions
        ( for(I, 0, N) do !A ^ elem(I) := I )
    ).

modifier(N, S) :-
    %$omp parallel do reduction(task, (+) : S)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for reduction modifiers
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + I ).

% An operator is any function the programmer promises is associative.
operator(N, S) :-
    %$omp parallel do reduction(add)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for reductions by operators other than +, *, min, max, /\, \/ and xor
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = add(S0, I) ).

:- func add(int, int) = int.
add(A, B) = A + B.

unknown(N, S) :-
    %$omp parallel do reduction((+) : T)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: T is not a value the loop carries from one iteration to the next, a fromto's Last or a state variable
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + I ).

several(N, S, P) :-
    %$omp parallel do reduction(+)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: a reduction clause without variables names the one scalar the loop carries from one iteration to the next, but it carries several
    ( for(I, 1, N), fromto(0, S0, S1, S), fromto(1, P0, P1, P) do
        S1 = S0 + I,
        P1 = P0 * I
    ).

% An array updated in place is not a reduction's.
none(N, !A) :-
    %$omp parallel do reduction(+)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: a reduction clause without variables names the one scalar the loop carries from one iteration to the next, but it carries none
    ( for(I, 0, N) do !A ^ elem(I) := I ).

twice(N, S) :-
    %$omp parallel do reduction((+) : S) reduction((*) : S)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: S is named in two reduction clauses
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + I ).

unsigned(N, S) :-
    %$omp parallel do reduction(+)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for reductions of type uint
    ( for(I, 1, N), fromto(0u, S0, S1, S) do S1 = S0 + uint.cast_from_int(I) ).

bitwise(N, S) :-
    %$omp parallel do reduction(xor)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: the reduction operator xor is not defined on floats
    ( for(I, 1, N), fromto(0.0, S0, S1, S) do S1 = S0 + float(I) ).

% A qualified operator is the type's own.
qualified(N, S) :-
    %$omp parallel do reduction(int.max)
% CHECK: omp-parallel-do-errors.m:[[@LINE-1]]:5: error: the reduction operator int.max is not defined on floats
    ( for(I, 1, N), fromto(0.0, S0, S1, S) do S1 = float.max(S0, float(I)) ).
