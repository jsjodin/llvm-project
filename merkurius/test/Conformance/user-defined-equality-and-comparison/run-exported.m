% MANUAL: User-defined equality and comparison
% An exported type with a user-defined equality, whose equality predicate
% is exported too: an importing module's unifications use it.
% MODULES: Inputs/exported_set.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/exported_set.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/exported_set.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:equal
% OUTPUT-NEXT:different
% OUTPUT-NOT:{{.}}

:- module run_exported.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exported_set, list.

:- pred test(set(int)::in, set(int)::in, io::di, io::uo) is det.
test(A, B, !IO) :-
    ( if A = B then
        io.write_string("equal\n", !IO)
    else
        io.write_string("different\n", !IO)
    ).

main(!IO) :-
    test(set([1, 2]), set([2, 1]), !IO),
    test(set([1, 2]), set([3]), !IO).
