% The checks of %$omp atomic (docs/INPROGRESS.md, Atomic updates of array
% elements): its goal updates one element of one array by an operator of
% int or float applied to the element's old value and an operand that
% does not use it; a capture also reads the element, before the update or
% after it, and gives that one value to the goals after it. The clauses
% other than update and capture are not yet implemented. Each error leaves
% its procedure out.
%
% RUN: not merkc -fc1 -fopenmp -emit-merk %s -o /dev/null 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module omp_atomic_errors.
:- interface.
:- import_module array.
:- pred no_operator(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred no_read(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred other_index(int::in, int::in, array(int)::array_di,
    array(int)::array_uo) is det.
:- pred other_constant(array(int)::array_di, array(int)::array_uo) is det.
:- pred element_twice(int::in, array(int)::array_di, array(int)::array_uo)
    is det.
:- pred divide(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred two_writes(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred gives_old(int::in, int::out, array(int)::array_di,
    array(int)::array_uo) is det.
:- pred captures_none(int::in, array(int)::array_di, array(int)::array_uo)
    is det.
:- pred captures_two(int::in, int::out, int::out, array(int)::array_di,
    array(int)::array_uo) is det.
:- pred strings(int::in, array(string)::array_di, array(string)::array_uo)
    is det.
:- pred read_clause(int::in, int::out, array(int)::in) is det.
:- pred seq_cst(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- implementation.
:- import_module int, string.

no_operator(I, !A) :-
    %$omp atomic
    !A ^ elem(I) := 1.
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic update must be !A ^ elem(I) := !.A ^ elem(I) op E, with op one of int's or float's +, -, * or int's /\, \/ and xor, and E not using the element

no_read(I, !A) :-
    %$omp atomic
    !A ^ elem(I) := I + 1.
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic update must be

other_index(I, J, !A) :-
    %$omp atomic
    !A ^ elem(I) := !.A ^ elem(J) + 1.
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic update must be

other_constant(!A) :-
    %$omp atomic
    !A ^ elem(0) := !.A ^ elem(1) + 1.
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic update must be

element_twice(I, !A) :-
    %$omp atomic
    !A ^ elem(I) := !.A ^ elem(I) + !.A ^ elem(I).
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic update must be

divide(I, !A) :-
    %$omp atomic
    !A ^ elem(I) := !.A ^ elem(I) // 2.
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic update must be

two_writes(I, !A) :-
    %$omp atomic
    ( !A ^ elem(I) := !.A ^ elem(I) + 1, !A ^ elem(I) := !.A ^ elem(I) + 1 ).
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic update must be

    % An update gives nothing but the array.
gives_old(I, Old, !A) :-
    %$omp atomic update
    ( Old = !.A ^ elem(I), !A ^ elem(I) := Old + 1 ).
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic update must be

captures_none(I, !A) :-
    %$omp atomic capture
    !A ^ elem(I) := !.A ^ elem(I) + 1.
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic capture must be an atomic update and a read of its element, V = !.A ^ elem(I) before it or V = !.A ^ elem(I) after it

captures_two(I, Old, New, !A) :-
    %$omp atomic capture
    ( Old = !.A ^ elem(I), !A ^ elem(I) := Old + 1, New = !.A ^ elem(I) ).
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic capture must be

strings(I, !A) :-
    %$omp atomic
    !A ^ elem(I) := !.A ^ elem(I) ++ "x".
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: an atomic update must be

read_clause(I, X, A) :-
    %$omp atomic read
    X = A ^ elem(I).
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for the read clause of the atomic directive

seq_cst(I, !A) :-
    %$omp atomic seq_cst
    !A ^ elem(I) := !.A ^ elem(I) + 1.
% CHECK: omp-atomic-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for the seq_cst clause of the atomic directive
