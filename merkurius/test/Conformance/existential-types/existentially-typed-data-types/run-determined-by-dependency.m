% MANUAL: Existentially typed data types
% An existentially quantified variable that occurs only in a constraint, in
% the range of a functional dependency of its class, is determined.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:"x of an int"
% OUTPUT-NOT:{{.}}

:- module run_determined_by_dependency.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass xable(T1, T2) <= (T1 -> T2) where [ func x(T1) = T2 ].

:- instance xable(int, string) where [ x(_) = "x of an int" ].

:- type t3
    --->    some [T1, T2] f5(T1) => xable(T1, T2).

:- pred show(t3::in, io::di, io::uo) is det.
show(f5(A), !IO) :-
    io.write_line(x(A), !IO).

main(!IO) :-
    show('new f5'(1), !IO).
