% MANUAL: pragma foreign_proc
% A det predicate, a semidet one, which sets SUCCESS_INDICATOR, and a
% function, each defined by a foreign_proc for C; a predicate of two modes
% with a foreign_proc for each.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:5
% OUTPUT-NEXT:3
% OUTPUT-NEXT:odd
% OUTPUT-NEXT:49
% OUTPUT-NEXT:2
% OUTPUT-NEXT:9
% OUTPUT-NOT:{{.}}

:- module run_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.
:- pred add(int::in, int::in, int::out) is det.
:- pragma foreign_proc("C", add(X::in, Y::in, Z::out),
    [promise_pure, will_not_call_mercury], "Z = X + Y;").
:- pred half(int::in, int::out) is semidet.
:- pragma foreign_proc("C", half(X::in, Y::out),
    [promise_pure, will_not_call_mercury],
    "SUCCESS_INDICATOR = (X % 2 == 0); if (SUCCESS_INDICATOR) Y = X / 2;").
:- func square(int) = int.
:- pragma foreign_proc("C", square(X::in) = (Y::out),
    [promise_pure, will_not_call_mercury], "Y = X * X;").
:- pred succ(int, int).
:- mode succ(in, out) is det.
:- mode succ(out, in) is det.
:- pragma foreign_proc("C", succ(X::in, Y::out),
    [promise_pure, will_not_call_mercury], "Y = X + 1;").
:- pragma foreign_proc("C", succ(X::out, Y::in),
    [promise_pure, will_not_call_mercury], "X = Y - 1;").

main(!IO) :-
    add(2, 3, A), io.write_int(A, !IO), io.nl(!IO),
    ( if half(6, H) then io.write_int(H, !IO) else io.write_string("odd", !IO) ),
    io.nl(!IO),
    ( if half(7, H7) then io.write_int(H7, !IO) else io.write_string("odd", !IO) ),
    io.nl(!IO),
    io.write_int(square(7), !IO), io.nl(!IO),
    succ(1, S), succ(P, 10), io.write_int(S, !IO), io.nl(!IO),
    io.write_int(P, !IO), io.nl(!IO).
