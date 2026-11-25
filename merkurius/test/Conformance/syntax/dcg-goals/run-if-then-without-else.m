% MANUAL: DCG-goals
% A DCG if-then without an else part, whose else is [] as in mmc. The
% manual requires the else part; the register records the difference.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[5]
% OUTPUT-NEXT:fails
% OUTPUT-NEXT:[3]
% OUTPUT-NOT:{{.}}

:- module run_if_then_without_else.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- pred p(list(int)::in, list(int)::out) is semidet.
p --> ( if [1] then [2] ).

main(!IO) :-
    ( if p([1, 2, 5], A) then io.write(A, !IO) else io.write_string("fails", !IO) ),
    io.nl(!IO),
    ( if p([1, 3], B) then io.write(B, !IO) else io.write_string("fails", !IO) ),
    io.nl(!IO),
    ( if p([3], C) then io.write(C, !IO) else io.write_string("fails", !IO) ),
    io.nl(!IO).
