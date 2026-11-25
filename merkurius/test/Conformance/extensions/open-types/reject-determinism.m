% DESIGN: Open terms (Prolog-style variables) / Operations, Checking rules
% Binding an open variable is semidet, as it may be bound already: so are
% full unification and taking apart an open term. A disjunction whose
% disjuncts bind a nonlocal open variable has a solution for each, as
% each may bind it differently. Construction into a variable that is still
% free is det.
% ORACLE: none
%
% RUN: not merkc -fc1 -fopen-types -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_determinism.
:- interface.
:- import_module list, open.
:- pred bind_det(open(int)::ia) is det.
% CHECK-DAG: reject-determinism.m:[[@LINE-1]]:9: error: the determinism declaration of bind_det(ia) is not satisfied: it is declared det, but inferred semidet
:- pred unify_det(open(int)::ia, open(int)::ia) is det.
% CHECK-DAG: reject-determinism.m:[[@LINE-1]]:9: error: the determinism declaration of unify_det(ia, ia) is not satisfied: it is declared det, but inferred semidet
:- pred take_apart(open(list(int))::ia, open(int)::oa) is det.
% CHECK-DAG: reject-determinism.m:[[@LINE-1]]:9: error: the determinism declaration of take_apart(ia, oa) is not satisfied: it is declared det, but inferred semidet
:- pred cases(open(list(int))::ia) is semidet.
% CHECK-DAG: reject-determinism.m:[[@LINE-1]]:9: error: the determinism declaration of cases(ia) is not satisfied: it is declared semidet, but inferred nondet
:- pred fresh_det(open(list(int))::oa) is det.
:- implementation.

bind_det(X) :-
    X = to_open(1).

unify_det(X, Y) :-
    X = Y.

take_apart(L, H) :-
    L = [H | _].

cases(L) :-
    ( L = []
    ; L = [_ | _]
    ).

    % Not an error: construction into a variable that is still free is det.
fresh_det(L) :-
    L = [_, _].
