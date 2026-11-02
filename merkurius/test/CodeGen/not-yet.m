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
    X = [1, 2],
    p(X).
% CHECK: not-yet.m:[[@LINE-2]]:9: error: not yet implemented: code generation for values of type list.list(int)

:- pred p(list(int)::in) is det.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for values of type list.list(int)
p(_).

:- pred q(int::in) is semidet.
q(X) :-
    ( X > 0 ; X < -5 ).
% CHECK: not-yet.m:[[@LINE-1]]:7: error: not yet implemented: code generation for disjunctions

:- pred r(T::in, T::out) is det.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for polymorphic procedures
r(X, X).

:- func w(int) = int.
w(X) = pow(X, 3).
% CHECK: not-yet.m:[[@LINE-1]]:8: error: not yet implemented: code generation for calls of int.pow/2

:- pred u(int::out) is multi.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for multi procedures
u(1).
u(2).
