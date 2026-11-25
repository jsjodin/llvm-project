% DESIGN: Loops / Types and purity
% A loop is typed in its clause by the constraints its translation would
% give: a counter and its bounds are ints, a foreach's list is a list of
% its element's type, and a fromto's four arguments have one type. An
% error names the iterator.
% ORACLE: none
%
% RUN: not merkc -fc1 -fdo-loops -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
:- module reject_type_errors.
:- interface.
:- import_module list.
:- pred p1(int::out) is det.
:- pred p2(list(string)::in, int::out) is det.
:- implementation.
:- import_module int.
p1(S) :- ( for(I, 1, 3.0), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: reject-type-errors.m:[[@LINE-1]]:12: error: type error in for(I, 1, 3.0): `3.0' has type float, but a counter and its bounds have type int
p2(Xs, S) :- ( foreach(X, Xs), fromto(0, S0, S1, S) do S1 = S0 + X ).
% CHECK: reject-type-errors.m:[[@LINE-1]]:{{[0-9]+}}: error:
