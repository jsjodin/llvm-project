% Loops compiled sequentially, as their translation (design document,
% "Loops"): each is a call of its loop predicate, a procedure whose mode
% is inferred and whose last call is the recursion. A for's trip count is
% counted down, so a loop to max_int ends, and strides and negative steps
% stop at the last value in range. The output is mmc's for the loops'
% translation written by hand, with the meaning's `I > Hi` test.
%
% RUN: merkc -fc1 -fdo-loops -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fdo-loops %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.O2 && %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK-LABEL: merk.proc private @loops.pi__loop_{{[0-9]+}}_7_5_0(
% MERK:         merk.call @loops.pi__body_{{[0-9]+}}_7_4_0(
% MERK:         merk.call @loops.pi__loop_{{[0-9]+}}_7_5_0(

% OUTPUT:3.1415927369231227
% OUTPUT-NEXT:65
% OUTPUT-NEXT:7.0
% OUTPUT-NEXT:10
% OUTPUT-NEXT:7
% OUTPUT-NEXT:4
% OUTPUT-NEXT:1
% OUTPUT-NEXT:3
% OUTPUT-NEXT:3
% OUTPUT-EMPTY:

:- module loops.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

main(!IO) :-
    pi(1000, P), io.write_float(P, !IO), io.nl(!IO),
    triangle(4, T), io.write_int(T, !IO), io.nl(!IO),
    array.init(4, 1.0, Y0),
    array.init(4, 2.0, X),
    saxpy(3.0, X, Y0, Y),
    io.write_float(Y ^ elem(3), !IO), io.nl(!IO),
    ( for(I, 10, 1, -3) do io.write_int(I, !IO), io.nl(!IO) ),
    ( for(I, max_int - 2, max_int), count(_, 1, C) do true ),
    io.write_int(C, !IO), io.nl(!IO),
    ( for(I, 1, 0) do io.write_int(I, !IO) ),
    ( for(I, min_int, min_int + 9, 4), fromto(0, S0, S1, S) do S1 = S0 + 1 ),
    io.write_int(S, !IO), io.nl(!IO).

:- pred pi(int::in, float::out) is det.
pi(N, Pi) :-
    Step = 1.0 / float(N),
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        X = (float(I) + 0.5) * Step,
        Y = 4.0 / (1.0 + X * X),
        S1 = S0 + Y
    ),
    Pi = Step * Sum.

:- pred triangle(int::in, int::out) is det.
triangle(N, Total) :-
    ( for(I, 1, N), fromto(0, T0, T1, Total) do
        ( for(J, 1, I), fromto(T0, U0, U1, T1) do U1 = U0 + I * J )
    ).

:- pred saxpy(float::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
saxpy(A, X, !Y) :-
    N = array.size(X),
    ( for(I, 0, N - 1) do
        !Y ^ unsafe_elem(I) := A * X ^ unsafe_elem(I) + !.Y ^ unsafe_elem(I)
    ).
