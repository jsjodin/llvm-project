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

:- pred q(int::in) is semidet.
q(X) :-
    ( X > 0 ; X < -5 ).
% CHECK: not-yet.m:[[@LINE-1]]:7: error: not yet implemented: code generation for disjunctions

:- some [T] pred r(T::out) is det.
% CHECK: not-yet.m:[[@LINE-1]]:18: error: not yet implemented: code generation for foreign_procs with existential types
:- pragma foreign_proc("C", r(X::out), [will_not_call_mercury, promise_pure], "X = 0;").

:- func w(int) = int.
w(X) = pow(X, 3).
% CHECK: not-yet.m:[[@LINE-1]]:8: error: not yet implemented: code generation for calls of int.pow/2

:- pred u(int::out) is multi.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for multi procedures
u(1).
u(2).

% An existentially quantified type variable inside an argument's type.
:- type exbox ---> some [T] exbox(list(T)).
:- func exlength(exbox) = int.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for values of type not_yet.exbox
exlength(exbox(L)) = list.length(L).
