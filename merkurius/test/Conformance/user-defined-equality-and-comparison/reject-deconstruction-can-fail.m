% MANUAL: User-defined equality and comparison
% A deconstruction of a value of a type with user-defined equality that
% can fail, other than an (in, in) unification, is an error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:39:{{[0-9]+}}: error:

:- module reject_deconstruction_can_fail.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.
:- type set(T)
    --->    set(list(T))
            where equality is set_equals.

:- pred set_equals(set(T)::in, set(T)::in) is semidet.
set_equals(S1, S2) :-
    subset(S1, S2),
    subset(S2, S1).

:- pred subset(set(T)::in, set(T)::in) is semidet.
subset(S1, S2) :-
    promise_equivalent_solutions [L1, L2] (
        S1 = set(L1),
        S2 = set(L2)
    ),
    all_in(L1, L2).

:- pred all_in(list(T)::in, list(T)::in) is semidet.
all_in([], _).
all_in([X | Xs], L) :-
    list.member(X, L),
    all_in(Xs, L).

:- pred is_empty(set(int)::in) is semidet.
is_empty(S) :-
    S = set([]).                                    % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
