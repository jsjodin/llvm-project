% Code generation reports the first construct of each procedure that it
% does not handle yet, and fails.
%
% RUN: not merkc -fc1 -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module not_yet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

main(!IO) :-
    X = [1, 2],
    p(X).
% CHECK: not-yet.m:[[@LINE-2]]:9: error: not yet implemented: code generation for values of type list.list(int)

:- pred p(list(int)::in) is det.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for values of type list.list(int)
p(_).

:- pred q(int::in, int::out) is semidet.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for semidet procedures
q(X, X).

:- pred r(T::in, T::out) is det.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for polymorphic procedures
r(X, X).

:- pred u(int::in, int::out) is det.
u(X, Y) :-
    ( if X = 0 then Y = 1 else Y = 2 ).
% CHECK: not-yet.m:[[@LINE-1]]:7: error: not yet implemented: code generation for goals of kind IfThenElse
