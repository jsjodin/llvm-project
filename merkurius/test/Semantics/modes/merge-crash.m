% A module that mmc 22.01.9 crashes on in mode analysis ("Software Error:
% predicate `hlds.instmap.merge_instmapping_delta_vars'/9: Unexpected:
% error merging var 7"), reduced from a module of
% generate-mode-programs.py --seed 9 (see the register). After the
% if-then-else, V2 has free parts in both branches' insts, and the
% disjunction unifies it, through V37, with V27, which has free parts
% too, which would alias them: a mode error.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s

:- module merge_crash.
:- interface.
:- import_module list.
:- type t ---> ta ; tb(int) ; tc(int, t).
:- pred p5(t::out, list(int)::in) is nondet.
:- implementation.
:- import_module int.
p5(H1, H2) :-
% CHECK: [[@LINE-1]]:1: warning: unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:
    V42 = 3,
    H1 = V41,
    V41 = tb(V42),
    V27 = tc(V39, V40),
    ( if H2 = [] then
        V6 = ta,
        V1 = V3,
        V2 = V6,
        V3 = [V4 | V5]
    else
        V14 = tc(V15, V16),
        V2 = V8,
        V13 = 4,
        V7 = [],
        V8 = tc(V9, V14),
        V1 = V7,
        V12 = 7,
        V11 = V12 + V13
    ),
    (
        V29 = tc(V30, V35)
    ;
        V27 = V37,
        V37 = V2,
% CHECK: [[@LINE-1]]:9: error: in the clause for p5(out, in): mode error in the unification of V37 and V2: V37 has inst unique(tc(free, free)), and V2 has inst unique(ta ; tc(free, unique(tc(free, free))))
        V38 = V1
    ).
