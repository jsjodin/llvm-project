% MANUAL: Adding foreign declarations
% foreign_decl's code is seen by the module's foreign_procs, whether local
% or not.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:80
% OUTPUT-NOT:{{.}}

:- module run_foreign_decl.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pragma foreign_decl("C", "#define SHARED_VALUE 40").
:- pragma foreign_decl("C", local, "static long twice(long x) { return 2 * x; }").
:- func value = int.
:- pragma foreign_proc("C", value = (X::out),
    [promise_pure, will_not_call_mercury],
    "X = twice(SHARED_VALUE);").

main(!IO) :- io.write_int(value, !IO), io.nl(!IO).
