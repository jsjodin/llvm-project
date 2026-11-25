% DESIGN: Open terms (Prolog-style variables) / Operations, Checking rules
% from_open, var, nonvar, open.write and open.string are impure, so a
% call of one is annotated `impure', and the predicate around it is
% declared impure or promised pure. A negation or an if-then-else
% condition that may bind a nonlocal open variable is in a promise_purity
% scope. A disjunction needs none, since each disjunct's bindings stay in
% its own solution.
% ORACLE: none
%
% RUN: not merkc -fc1 -fopen-types -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_purity.
:- interface.
:- import_module io, list, open.
:- pred unannotated(open(list(int))::ia, list(int)::out) is semidet.
:- pred undeclared(open(list(int))::ia, list(int)::out) is semidet.
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:9: error: pred reject_purity.undeclared/2 is impure, so it must be declared `impure' or promised pure
:- pred negation(open(int)::ia) is semidet.
:- pred condition(open(int)::ia, int::out) is det.
:- pred var_call(open(int)::ia) is semidet.
:- pred nonvar_call(open(int)::ia) is semidet.
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:9: error: pred reject_purity.nonvar_call/1 is impure, so it must be declared `impure' or promised pure
:- pred write_call(open(int)::ia, io::di, io::uo) is det.
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:9: error: pred reject_purity.write_call/3 is impure, so it must be declared `impure' or promised pure
:- pred string_call(open(int)::ia, string::out) is det.
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:9: error: pred reject_purity.string_call/2 is impure, so it must be declared `impure' or promised pure
:- pred disjunction(open(int)::ia, int::out) is multi.
:- implementation.
:- import_module int.

unannotated(X, L) :-
    promise_pure ( from_open(X, L) ).
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:20: error: the call of impure pred open.from_open/2 must be annotated `impure'

undeclared(X, L) :-
    impure from_open(X, L).

negation(X) :-
    \+ X = to_open(1).
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:5: error: in the clause for negation(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal X has inst any and appears in its body

condition(X, N) :-
    ( if X = to_open(1) then N = 1 else N = 0 ).
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:7: error: in the clause for condition(ia, out): purity error: the if-then-else must be inside a promise_purity scope, because nonlocal X has inst any and appears in its condition

var_call(X) :-
    promise_pure ( var(X) ).
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:20: error: the call of impure pred open.var/1 must be annotated `impure'

nonvar_call(X) :-
    impure nonvar(X).

write_call(X, !IO) :-
    open.write(X, !IO).
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:5: error: the call of impure pred open.write/3 must be annotated `impure'

string_call(X, S) :-
    S = open.string(X).
% CHECK-DAG: reject-purity.m:[[@LINE-1]]:9: error: impure func open.string/1 must be applied in a unification of its own, annotated `impure'

disjunction(X, N) :-
    ( X = to_open(1), N = 1
    ; N = 2
    ).
