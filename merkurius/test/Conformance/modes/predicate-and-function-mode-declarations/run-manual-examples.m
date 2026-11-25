% MANUAL: Predicate and function mode declarations
% A predicate or function may have several modes, each a procedure, and a
% function may be reversible. with_inst gives a declared higher-order inst
% its arguments. A single mode may be combined with the type declaration.
% A function without a mode declaration has the default mode, its
% arguments in and its result out. A call with an argument more bound
% than its mode needs uses an implied mode.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[1, 2, 3]
% OUTPUT-NEXT:[{[], [1, 2]}, {[1], [2]}, {[1, 2], []}]
% OUTPUT-NEXT:5
% OUTPUT-NEXT:4
% OUTPUT-NEXT:6
% OUTPUT-NEXT:24
% OUTPUT-NEXT:5
% OUTPUT-NEXT:3
% OUTPUT-NEXT:42
% OUTPUT-NEXT:five is 5
% OUTPUT-NEXT:five is not 6
% OUTPUT-NOT:{{.}}

:- module run_manual_examples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- pred app(list(T), list(T), list(T)).
:- mode app(in, in, out) is det.
:- mode app(out, out, in) is multi.
app([], Ys, Ys).
app([X | Xs], Ys, [X | Zs]) :-
    app(Xs, Ys, Zs).

:- func succ(int) = int.
:- mode succ(in) = out is det.
:- mode succ(out) = in is det.
succ(X) = X + 1.

:- inst adder_pred == (pred(in, in, out) is det).
:- inst adder_func == (func(in, in) = out is det).
:- type adder_pred == pred(int, int, int).
:- type adder_func == (func(int, int) = int).

:- pred add3(int) `with_type` adder_pred.
:- mode add3(in) `with_inst` adder_pred.
add3(A, B, C, A + B + C).

:- func mul3(int) `with_type` adder_func.
:- mode mul3(in) `with_inst` adder_func.
mul3(A, B, C) = A * B * C.

:- pred sub(int::in) `with_type` adder_pred `with_inst` adder_pred.
sub(A, B, C, A - B - C).

:- func len(list(T)::in) = (int::out) is det.
len(L) = list.length(L).

:- func double(int) = int.
double(X) = X * 2.

:- pred five(int::out) is det.
five(5).

main(!IO) :-
    app([1, 2], [3], L),
    io.write_line(L, !IO),
    Splits = solutions_list([1, 2]),
    io.write_line(Splits, !IO),
    io.write_line(succ(4), !IO),
    5 = succ(P),
    io.write_line(P, !IO),
    add3(1, 2, 3, A),
    io.write_line(A, !IO),
    io.write_line(mul3(2, 3, 4), !IO),
    sub(10, 2, 3, S),
    io.write_line(S, !IO),
    io.write_line(len([a, b, c]), !IO),
    io.write_line(double(21), !IO),
    ( if five(5) then
        io.write_string("five is 5\n", !IO)
    else
        io.write_string("five is not 5\n", !IO)
    ),
    ( if five(6) then
        io.write_string("five is 6\n", !IO)
    else
        io.write_string("five is not 6\n", !IO)
    ).

:- func solutions_list(list(int)) = list({list(int), list(int)}).
solutions_list(L) = Splits :-
    solutions.solutions(
        (pred({Xs, Ys}::out) is multi :- app(Xs, Ys, L)), Splits).
