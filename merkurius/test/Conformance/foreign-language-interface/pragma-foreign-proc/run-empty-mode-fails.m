% MANUAL: pragma foreign_proc
% A semidet mode with neither clauses nor a foreign_proc, of a predicate
% whose other mode has a foreign_proc, fails, as mmc infers it.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1
% OUTPUT-NEXT:failed
% OUTPUT-NOT:{{.}}

:- module run_empty_mode_fails.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p(int, int).
:- mode p(in, out) is det.
:- mode p(out, in) is semidet.
:- pragma foreign_proc("C", p(X::in, Y::out),
    [promise_pure, will_not_call_mercury], "Y = X;").

main(!IO) :-
    p(1, Y), io.write_int(Y, !IO), io.nl(!IO),
    ( if p(X, 3) then io.write_int(X, !IO) else io.write_string("failed", !IO) ),
    io.nl(!IO).
