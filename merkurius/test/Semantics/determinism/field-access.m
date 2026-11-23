% The determinism of field accesses and updates (reference manual, "Field
% access functions"): each fails when the term's constructor is not the
% field's, as the unification it stands for does, so it is det when the
% type has only that constructor and semidet when it has others. Where
% the term is known to have the field's constructor, after a switch on it
% or by its inst, it cannot fail.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module field_access.
:- interface.

:- type one ---> one(a :: int, b :: int).
:- type two ---> two(c :: int) ; other(d :: int) ; none.

:- pred get_one(one::in, int::out) is det.
:- pred set_one(one::in, one::out) is det.
:- pred get_two(two::in, int::out) is semidet.
:- pred set_two(two::in, two::out) is semidet.
% CHECK-DAG: field-access.m:[[@LINE+1]]:9: error: the determinism declaration of get_det(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred get_det(two::in, int::out) is det.
% CHECK-DAG: field-access.m:[[@LINE+1]]:9: error: the determinism declaration of set_det(in, out) is not satisfied: it is declared det, but inferred semidet
:- pred set_det(two::in, two::out) is det.
:- pred known_switch(two::in, int::out) is det.
:- pred known_inst(int::out) is det.
:- pred known_update(two::out) is det.

:- implementation.

get_one(X, X ^ a).
set_one(X, X ^ b := 1).
get_two(X, X ^ c).
set_two(X, X ^ d := 2).
get_det(X, X ^ c).
% CHECK-DAG: field-access.m:[[@LINE-1]]:12: because: the call to c(in, out) can fail
set_det(X, X ^ d := 2).
% CHECK-DAG: field-access.m:[[@LINE-1]]:12: because: the call to d :=(in, in, out) can fail
known_switch(X, N) :-
    (
        X = two(_),
        N = X ^ c
    ;
        X = other(_),
        N = X ^ d
    ;
        X = none,
        N = 0
    ).
known_inst(X ^ c) :-
    X = two(3).
known_update(X0 ^ c := 2) :-
    X0 = two(1).
