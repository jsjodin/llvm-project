% Code generation reports the first construct of each procedure that it
% does not handle yet, and fails.
%
% RUN: not merkc -fc1 -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module not_yet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    P = (pred(Y::out) is det :- Y = 1),
    p(P).
% CHECK: not-yet.m:[[@LINE-2]]:10: error: not yet implemented: code generation for values of type pred(int)

:- pred p(pred(int)::in(pred(out) is det)) is det.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for values of type pred(int)
p(_).

% A type with user-defined equality, whose unifications call its
% predicate.
:- type set ---> set(list(int)) where equality is equal.
:- pred equal(set::in, set::in) is semidet.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for values of type not_yet.set
equal(set(X), set(X)).

% A unification of strings in a term.
:- pred same_name(named::in, named::in) is semidet.
:- type named ---> named(string).
same_name(X, X).
% CHECK: not-yet.m:[[@LINE-1]]:14: error: not yet implemented: code generation for this unification

:- pred q(int::in) is semidet.
q(X) :-
    ( X > 0 ; X < -5 ).
% CHECK: not-yet.m:[[@LINE-1]]:7: error: not yet implemented: code generation for disjunctions

:- pred r(list(T)::in, list(T)::out) is det.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for values of type list.list(T0)
r(X, X).

:- func w(int) = int.
w(X) = pow(X, 3).
% CHECK: not-yet.m:[[@LINE-1]]:8: error: not yet implemented: code generation for calls of int.pow/2

:- pred u(int::out) is multi.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for multi procedures
u(1).
u(2).
