% MANUAL: Instance declarations
% The manual's instance of foo(int): a method named by an existing
% function, one defined by a fact, one by a rule, one by a DCG rule and one
% by several clauses.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:7
% OUTPUT-NEXT:4
% OUTPUT-NEXT:5
% OUTPUT-NEXT:17
% OUTPUT-NEXT:{0, -9}
% OUTPUT-NOT:{{.}}

:- module run_manual_foo.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, int.

:- typeclass foo(T) where [
    func method1(T, T) = int,
    func method2(T) = int,
    pred method3(T::in, int::out) is det,
    pred method4(T::in, io.state::di, io.state::uo) is det,
    func method5(bool, T) = T
].

:- instance foo(int) where [
    % method defined by naming the implementation
    func(method1/2) is (+),

    % method defined by a fact
    method2(X) = X + 1,

    % method defined by a rule
    (method3(X, Y) :- Y = X + 2),

    % method defined by a DCG rule
    (method4(X) --> io.print(X), io.nl),

    % method defined by multiple clauses
    method5(no, _) = 0,
    (method5(yes, X) = Y :- X + Y = 0)
].

main(!IO) :-
    io.write_line(method1(3, 4), !IO),
    io.write_line(method2(3), !IO),
    method3(3, Z),
    io.write_line(Z, !IO),
    method4(17, !IO),
    io.write_line({method5(no, 9), method5(yes, 9)}, !IO).
