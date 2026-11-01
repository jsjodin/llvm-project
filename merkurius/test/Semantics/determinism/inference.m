% Determinism inference (reference manual, "Determinism checking and
% inference"), as mmc infers it: the determinisms of the procedures
% declared without one are the least fixpoint of the rules, starting from
% erroneous, where the manual says det (see the design document's
% register), so that a recursion that never returns is erroneous. Such a
% procedure's body is in a context where all its solutions are wanted, so
% that a committed choice call in it is an error, and a declared caller
% gives its calls' reasons as for a declared callee.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module inference.
:- interface.
:- type t ---> a ; b ; c(int).
% ERR: inference.m:[[@LINE+1]]:9: error: the determinism declaration of top(in, out) is not satisfied: it is declared det, but inferred nondet
:- pred top(int::in, int::out) is det.
:- implementation.
:- import_module int, list, require.

% The reasons for a declared caller are its calls to inferred callees.
% ERR: inference.m:[[@LINE+2]]:27: because: the call to u_semi(in, out) can fail
% ERR: inference.m:[[@LINE+1]]:41: because: the call to u_multi(out) can succeed more than once
top(X, Y) :- u_det(X, A), u_semi(A, B), u_multi(C), Y = B + C + u_f(X).

:- pred m(int::out) is multi.
m(1).
m(2).
:- pred cm(int::out) is cc_multi.
cm(1).
cm(2).

% CHECK-LABEL: u_det(in, out), inferred det
:- pred u_det(int::in, int::out).
u_det(X, X + 1).
% CHECK-LABEL: u_semi(in, out), inferred semidet
:- pred u_semi(int::in, int::out).
u_semi(X, Y) :- X > 0, Y = X.
% CHECK-LABEL: u_multi(out), inferred multi
:- pred u_multi(int::out).
u_multi(Y) :- m(Y).
% CHECK-LABEL: u_nondet(in, out), inferred nondet
:- pred u_nondet(int::in, int::out).
u_nondet(X, Y) :- X > 0, m(Y).
% CHECK-LABEL: u_fail(out), inferred failure
:- pred u_fail(int::out).
u_fail(_) :- fail.
% CHECK-LABEL: u_err(out), inferred erroneous
:- pred u_err(int::out).
u_err(_) :- error("u_err").
% CHECK-LABEL: u_sw(in, out), inferred semidet
:- pred u_sw(t::in, int::out).
u_sw(a, 1).
u_sw(b, 2).
% CHECK-LABEL: u_g(in) = out, inferred semidet
:- func u_g(int) = int.
:- mode u_g(in) = out.
u_g(X) = Y :- X > 0, Y = X.

% Recursion: one that never returns is erroneous; a det one is det, as is
% one through a callee inferred later.
% CHECK-LABEL: u_loop(in), inferred erroneous
:- pred u_loop(int::in).
u_loop(X) :- u_loop(X).
% CHECK-LABEL: u_loop2(in, out), inferred erroneous
:- pred u_loop2(int::in, int::out).
u_loop2(X, Y) :- u_loop2(X, Y), X > 3.
% CHECK-LABEL: u_rec(in, out), inferred det
:- pred u_rec(list(int)::in, int::out).
u_rec([], 0).
u_rec([H | T], S) :- u_rec(T, S0), S = S0 + H.
% CHECK-LABEL: u_deep(in, out), inferred multi
:- pred u_deep(int::in, int::out).
u_deep(X, Y) :-
    ( if X = 0 then Y = 0 else u_deep(X - 1, Y0), u_multi(Z), Y = Y0 + Z ).

% Mutual recursion: one that can only fail, and one that can fail or
% succeed.
% CHECK-LABEL: u_fl(in, out), inferred failure
:- pred u_fl(int::in, int::out).
u_fl(X, Y) :- u_fl2(X, Y).
% CHECK-LABEL: u_fl2(in, out), inferred failure
:- pred u_fl2(int::in, int::out).
u_fl2(X, Y) :- X = 0, u_fl(X, Y).
% CHECK-LABEL: u_even(in), inferred semidet
:- pred u_even(int::in).
u_even(X) :- ( X = 0 ; X > 0, u_odd(X - 1) ).
% CHECK-LABEL: u_odd(in), inferred semidet
:- pred u_odd(int::in).
u_odd(X) :- X > 0, u_even(X - 1).

% Committed choice: all the solutions of an undeclared procedure are
% wanted, so a cc_multi call is an error there, and multi.
% CHECK-LABEL: u_cc(out), inferred multi
:- pred u_cc(int::out).
% ERR: inference.m:[[@LINE+1]]:12: error: the call to cm(out) with determinism cc_multi is in a context that requires all solutions
u_cc(Y) :- cm(Y).
% CHECK-LABEL: u_cc2(out), inferred nondet
:- pred u_cc2(int::out).
% ERR: inference.m:[[@LINE+2]]:13: error: the call to cm(out) with determinism cc_multi is in a context that requires all solutions
% ERR: inference.m:[[@LINE+1]]:20: because: the call to >(in, in) can fail
u_cc2(Y) :- cm(Y), Y > 0.
% Unless a scope promises its solutions are equivalent.
% CHECK-LABEL: u_pes(out), inferred det
:- pred u_pes(int::out).
u_pes(Y) :- promise_equivalent_solutions [Y] cm(Y).

% The primary mode of a function cannot be multi, by its implicit
% declaration.
% ERR: inference.m:[[@LINE+3]]:9: error: invalid determinism for u_f(in) = out: the primary mode of a function cannot be multi
% ERR: inference.m:[[@LINE+2]]:9: error: the determinism declaration of u_f(in) = out is not satisfied: it is declared det, but inferred multi
% ERR: inference.m:[[@LINE+2]]:15: because: the call to m(out) can succeed more than once
:- func u_f(int) = int.
u_f(X) = Y :- m(Y0), Y = X + Y0.

:- pred uses(int::in) is semidet.
uses(X) :-
    u_nondet(X, _), u_fail(_), u_err(_), u_sw(a, _), _ = u_g(X),
    u_loop(X), u_loop2(X, _), u_rec([], _), u_deep(X, _), u_fl(X, _),
    u_even(X), u_cc(_), u_cc2(_), u_pes(_).
