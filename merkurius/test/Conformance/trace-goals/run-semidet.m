% MANUAL: Trace goals
% A trace goal in a semidet predicate, which runs before a test that
% succeeds.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:3
% OUTPUT-NEXT:yes
% OUTPUT-NOT:{{.}}

:- module run_semidet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in) is semidet.
p(X) :- trace [io(!S)] ( io.write_int(X, !S), io.nl(!S) ), X > 1.
main(!IO) :- ( if p(3) then io.write_string("yes\n", !IO) else true ).
