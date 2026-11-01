% Bound insts with a uniqueness, unique(...) (reference manual,
% "Destructive update"), and mostly_unique(...), which mmc has too though
% the manual does not (see the design document's register), with unique
% insts below their top level. mmc accepts them, though the manual says
% it cannot use nested unique modes ("Limitations of the current
% implementation").
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: %s < %t.err
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module nested_unique.
:- interface.
:- type t ---> t(int).
:- type p ---> p(t, t).
:- type e ---> a ; b(t).

:- inst up == unique(p(unique, unique)).
% CHECK: [[@LINE-1]]:9 inst nested_unique.up/0 == unique(p(unique, unique))
:- inst mup == mostly_unique(p(mostly_unique, unique)).
% CHECK: [[@LINE-1]]:9 inst nested_unique.mup/0 == mostly_unique(p(mostly_unique, unique))
:- inst bu == bound_unique(a ; b(unique)).
% CHECK: [[@LINE-1]]:9 inst nested_unique.bu/0 == unique(a ; b(unique))
:- inst ue == unique((a ; b(mostly_unique))).
% CHECK: [[@LINE-1]]:9 inst nested_unique.ue/0 == unique(a ; b(mostly_unique))
:- mode dip == up >> dead.
:- mode uop == free >> up.
:- mode mdip == mup >> mostly_dead.

:- pred update_first(p::dip, p::uop) is det.
:- pred pair(t::di, t::di, p::uop) is det.
:- pred first(p::dip, t::uo) is det.
:- pred update_both(p::dip, p::uop) is det.
:- pred mostly_first(p::mdip, t::uo) is det.
:- pred mostly_update(p::mdip, t::uo) is det.
:- pred in_b(e::in(bu), e::out(ue)) is det.

:- implementation.
:- import_module int.

:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).

update_first(p(A0, B), p(A, B)) :- updu(1, A0, A).
% CHECK: pred nested_unique.update_first/2 mode 0:
% CHECK: final: HeadVar__1 unique(p(unique, unique)), HeadVar__2 unique(p(unique, unique))
pair(A, B, p(A, B)).
% CHECK: pred nested_unique.pair/3 mode 0:
% CHECK: final: A unique, B unique, HeadVar__3 unique(p(unique, unique))
first(p(A, _), A).
% CHECK: pred nested_unique.first/2 mode 0:
% CHECK: final: HeadVar__1 unique(p(unique, unique)), A unique
update_both(P0, P) :-
    P0 = p(A0, B0), updu(1, A0, A), updu(2, B0, B), P = p(A, B).
% CHECK: pred nested_unique.update_both/2 mode 0:
% CHECK: final: P0 unique(p(unique, unique)), P unique(p(unique, unique))
mostly_first(p(A, _), A).
% ERR-DAG: nested-unique.m:[[@LINE-1]]:14: error: in the clause for mostly_first((nested_unique.mup >> builtin.mostly_dead), uo): mode error: argument 2, A, did not get sufficiently instantiated: its final inst is mostly_unique, but unique is expected
mostly_update(P0, P) :- P0 = p(A0, _), updu(1, A0, P).
% ERR-DAG: nested-unique.m:[[@LINE-1]]:40: error: in the clause for mostly_update((nested_unique.mup >> builtin.mostly_dead), uo): mode error in argument 2 of the call to pred nested_unique.updu/3: A0 has inst mostly_unique, but unique is expected
% A unique input passed on is shared, below a named inst too.
in_b(E, E).
% ERR-DAG: nested-unique.m:[[@LINE-1]]:9: error: in the clause for in_b(in(nested_unique.bu), out(nested_unique.ue)): mode error: argument 1, E, did not get sufficiently instantiated: its final inst is bound(a ; b(ground)), but nested_unique.bu is expected
