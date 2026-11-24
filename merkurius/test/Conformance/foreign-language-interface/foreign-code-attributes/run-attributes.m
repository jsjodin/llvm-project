% MANUAL: Foreign code attributes
% Each attribute of the manual's list, and an impure foreign_proc, whose
% calls are not merged, with a C static variable.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[100, 101, 102]
% OUTPUT-NEXT:49
% OUTPUT-NEXT:27
% OUTPUT-NOT:{{.}}

:- module run_attributes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.
:- impure pred next(int::out) is det.
:- pragma foreign_proc("C", next(N::out),
    [will_not_call_mercury, may_not_duplicate],
    "static long n = 0; N = 100 + n++;").
:- func square(int) = int.
:- pragma foreign_proc("C", square(X::in) = (Y::out),
    [promise_pure, will_not_call_mercury, thread_safe, may_duplicate,
    does_not_affect_liveness, terminates, will_not_modify_trail,
    tabled_for_io, may_export_body, will_not_throw_exception,
    will_not_call_mm_tabled],
    "Y = X * X;").
:- semipure func cube(int) = int.
:- pragma foreign_proc("C", cube(X::in) = (Y::out),
    [promise_semipure, may_call_mercury, not_thread_safe, does_not_terminate,
    may_call_mm_tabled, affects_liveness, may_not_export_body],
    "Y = X * X * X;").
:- pred count3(list(int)::out) is det.
count3([A, B, C]) :-
    promise_pure ( impure next(A), impure next(B), impure next(C) ).

main(!IO) :-
    count3(L), io.write(L, !IO), io.nl(!IO),
    io.write_int(square(7), !IO), io.nl(!IO),
    promise_pure ( semipure C = cube(3) ),
    io.write_int(C, !IO), io.nl(!IO).
