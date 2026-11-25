% MANUAL: Adding foreign definitions
% foreign_code defines a C variable and function, which a foreign_decl
% declares and foreign_procs use.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:100
% OUTPUT-NEXT:81
% OUTPUT-NOT:{{.}}

:- module run_foreign_code.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pragma foreign_code("C",
    "long counter_base = 100; long sq(long x) { return x * x; }").
:- pragma foreign_decl("C", local, "extern long counter_base; long sq(long);").
:- func base = int.
:- pragma foreign_proc("C", base = (X::out),
    [promise_pure, will_not_call_mercury], "X = counter_base;").
:- func square(int) = int.
:- pragma foreign_proc("C", square(X::in) = (Y::out),
    [promise_pure, will_not_call_mercury], "Y = sq(X);").

main(!IO) :- io.write_int(base, !IO), io.nl(!IO), io.write_int(square(9), !IO), io.nl(!IO).
