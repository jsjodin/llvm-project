% Determinism as mmc finds it, from the liveness of mode analysis, before
% it deletes the assignments of variables not used after them. In a cast
% whose variable an assignment deleted later still uses, the variable is
% bound, and not an extra one; only the deleted assignment's own cast
% lists one. A unification with a term that binds the term's free
% arguments cannot fail, even where mode analysis built the term first.
% Of a committed choice call in a goal with no outputs, whose first
% solution is the only one wanted, the reasons that all are needed, when
% something in the goal asks for them, are the goals after it that can
% fail too, as in mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module before_deletion.
:- interface.
:- type t ---> t(int).
:- type p ---> p(t, t).
:- inst up == unique(p(ground, ground)).
:- mode gdi == (ground >> clobbered).
:- pred cast_kept(int::in, int::out) is cc_multi.
:- pred cast_deleted(int::in, int::out) is det.
:- pred field_bound(t::gdi, t::out) is det.
:- pred fields_free(t::gdi, t::out) is det.
:- implementation.
:- import_module int.
:- pred m(int::out) is multi.
m(1).
m(2).
:- func mk(int) = t.
mk(N) = t(N).
:- pred cp(t::gdi, t::out) is det.
cp(T, T).
:- pred pfst(p::(up >> dead), t::out) is det.
pfst(p(A, _), A).

cast_kept(A, B) :-
    ( promise_equivalent_solutions [C] ( 3 > 1, C = A ), D = C ; D = A ),
    promise_equivalent_solutions [E] ( E = D ),
% ERR-DAG: before-deletion.m:[[@LINE-1]]:5: error: the promise_equivalent_solutions goal lists an extra variable: E
    m(B).
cast_deleted(A, B) :-
    ( promise_equivalent_solutions [C] ( 3 > 1, C = A ), D = C ; D = A ),
% ERR-DAG: before-deletion.m:[[@LINE-1]]:7: error: the promise_equivalent_solutions goal lists an extra variable: C
    B = 1.
field_bound(T0, T) :-
    cp(T0, T3),
    P5 = p(T3, mk(2)), P5 = p(T4, _),
    P9 = p(T0, mk(3)), pfst(P9, T).
fields_free(T0, T) :-
    cp(T0, T3),
    P5 = p(T3, t(2)), P5 = p(_, _),
    P9 = p(T0, t(3)), pfst(P9, T).
