% array.generate, generate_foldl and generate_foldl2, and the foldl family,
% which the library compiles from array.m rather than code generation
% building them in: arrays of ints, floats, strings and terms; closures
% with captured values; det and semidet modes, failing at the first element
% and later, and io as the accumulator, which shows the order of the calls.
% The output is mmc's. array-errors.m checks generate's negative sizes.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% With -fmmc-uniqueness, the module numbers array's procedures by array.m's
% modes, not the sound stub's that the library has, so it cannot call them.
% RUN: not merkc -fc1 -emit-merk -fmmc-uniqueness %s 2>&1 | FileCheck %s --check-prefix=MMC-UNIQUENESS
% MMC-UNIQUENESS: error: not yet implemented: code generation for calls of array.generate/2
%
% OUTPUT:array([])
% OUTPUT-NEXT:array([0, 1, 4, 9, 16])
% OUTPUT-NEXT:array([circle(0.0), rect(1, 2), circle(1.0)])
% OUTPUT-NEXT:array([7, 8, 9, 10])
% OUTPUT-NEXT:{array(["0", "10", "20", "30"]), 6}
% OUTPUT-NEXT:{array([]), 5}
% OUTPUT-NEXT:making 0
% OUTPUT-NEXT:making 1
% OUTPUT-NEXT:making 2
% OUTPUT-NEXT:array([100, 99, 98])
% OUTPUT-NEXT:{array([0, 2, 4]), 7}
% OUTPUT-NEXT:fails
% OUTPUT-NEXT:{array([0.5, 1.5, 2.5, 3.5]), 27, "s0123"}
% OUTPUT-NEXT:31
% OUTPUT-NEXT:[6, 2, 9, 5, 1, 4, 1, 3]
% OUTPUT-NEXT:42
% OUTPUT-NEXT:[12, 4, 18, 10, 2, 8, 2, 6]
% OUTPUT-NEXT:3 1 4 1 5 9 2 6 
% OUTPUT-NEXT:17.0
% OUTPUT-NEXT:">abcdef"
% OUTPUT-NEXT:odd found
% OUTPUT-NEXT:12
% OUTPUT-NEXT:odd found
% OUTPUT-NEXT:{3, [6, 4, 2]}
% OUTPUT-NEXT:{10, 24}
% OUTPUT-NEXT:1:1 2:3 3:6 4:10 
% OUTPUT-NEXT:10
% OUTPUT-NEXT:{10, -10, "1234"}
% OUTPUT-NEXT:{10, 20, 30, 40}
% OUTPUT-NEXT:1234
% OUTPUT-NEXT:{10, 30, 4, -10}
% OUTPUT-NEXT:too big

:- module array_folds.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, string.

:- type shape
    --->    circle(float)
    ;       rect(int, int).

main(!IO) :-
    generate(!IO),
    folds(!IO),
    semidet_folds(!IO),
    more_folds(!IO).

:- func square(int) = int.
square(I) = I * I.

:- func shape(int) = shape.
shape(I) = ( if I mod 2 = 0 then circle(float(I) / 2.0) else rect(I, I + 1) ).

:- pred number(int::in, string::out, int::in, int::out) is det.
number(I, S, N0, N) :-
    S = string.int_to_string(I * 10),
    N = N0 + I.

:- pred number_io(int::in, int::out, io::di, io::uo) is det.
number_io(I, X, !IO) :-
    io.write_string(string.format("making %d\n", [i(I)]), !IO),
    X = 100 - I.

:- pred small(int::in, int::out, int::in, int::out) is semidet.
small(I, X, N0, N) :-
    I < 3,
    X = I * 2,
    N = N0 + X.

:- pred two(int::in, float::out, int::in, int::out, string::in, string::out)
    is det.
two(I, X, N0, N, S0, S) :-
    X = float(I) + 0.5,
    N = N0 * 2 + I,
    S = S0 ++ string.int_to_string(I).

:- pred generate(io::di, io::uo) is det.
generate(!IO) :-
    io.write_line(array.generate(0, square), !IO),
    io.write_line(array.generate(5, square), !IO),
    io.write_line(array.generate(3, shape), !IO),
    Base = 7,
    io.write_line(array.generate(4, (func(I) = I + Base)), !IO),
    array.generate_foldl(4, number, A1, 0, N1),
    io.write_line({A1, N1}, !IO),
    array.generate_foldl(0, number, A2, 5, N2),
    io.write_line({A2, N2}, !IO),
    array.generate_foldl(3, number_io, A3, !IO),
    io.write_line(A3, !IO),
    ( if array.generate_foldl(3, small, A4, 1, N4) then
        io.write_line({A4, N4}, !IO)
    else
        io.write_string("fails\n", !IO)
    ),
    ( if array.generate_foldl(5, small, A5, 1, N5) then
        io.write_line({A5, N5}, !IO)
    else
        io.write_string("fails\n", !IO)
    ),
    array.generate_foldl2(4, two, A6, 1, N6, "s", S6),
    io.write_line({A6, N6, S6}, !IO).

:- pred folds(io::di, io::uo) is det.
folds(!IO) :-
    A = array.from_list([3, 1, 4, 1, 5, 9, 2, 6]),
    Empty = array.make_empty_array : array(int),
    io.write_line(array.foldl((func(X, S) = S + X), A, 0), !IO),
    io.write_line(array.foldl((func(X, L) = [X | L]), A, []), !IO),
    io.write_line(array.foldl((func(X, S) = S + X), Empty, 42), !IO),
    array.foldl((pred(X::in, L0::in, L::out) is det :- L = [X * 2 | L0]),
        A, [], L1),
    io.write_line(L1, !IO),
    array.foldl((pred(X::in, !.IO::di, !:IO::uo) is det :-
            io.write_int(X, !IO), io.write_char(' ', !IO)),
        A, !IO),
    io.nl(!IO),
    Shapes = array.generate(4, shape),
    array.foldl(area, Shapes, 0.0, Area),
    io.write_line(Area, !IO),
    Strings = array.from_list(["a", "bc", "def"]),
    io.write_line(array.foldl((func(S, Acc) = Acc ++ S), Strings, ">"), !IO).

:- pred area(shape::in, float::in, float::out) is det.
area(circle(R), A, A + 3.0 * R * R).
area(rect(W, H), A, A + float(W * H)).

:- pred semidet_folds(io::di, io::uo) is det.
semidet_folds(!IO) :-
    A = array.from_list([2, 4, 6, 7, 8]),
    ( if array.foldl(even_sum, A, 0, S1) then
        io.write_line(S1, !IO)
    else
        io.write_string("odd found\n", !IO)
    ),
    B = array.from_list([2, 4, 6]),
    ( if array.foldl(even_sum, B, 0, S2) then
        io.write_line(S2, !IO)
    else
        io.write_string("odd found\n", !IO)
    ),
    ( if array.foldl2(even_count, A, 0, C3, [], L3) then
        io.write_line({C3, L3}, !IO)
    else
        io.write_string("odd found\n", !IO)
    ),
    ( if array.foldl2(even_count, B, 0, C4, [], L4) then
        io.write_line({C4, L4}, !IO)
    else
        io.write_string("odd found\n", !IO)
    ).

:- pred even_sum(int::in, int::in, int::out) is semidet.
even_sum(X, S0, S) :-
    X mod 2 = 0,
    S = S0 + X.

:- pred even_count(int::in, int::in, int::out, list(int)::in, list(int)::out)
    is semidet.
even_count(X, C0, C, L0, L) :-
    X mod 2 = 0,
    C = C0 + 1,
    L = [X | L0].

:- pred more_folds(io::di, io::uo) is det.
more_folds(!IO) :-
    A = array.from_list([1, 2, 3, 4]),
    array.foldl2(sum_product, A, 0, S, 1, P),
    io.write_line({S, P}, !IO),
    array.foldl2((pred(X::in, S0::in, S1::out, !.IO::di, !:IO::uo) is det :-
            S1 = S0 + X, io.write_string(string.format("%d:%d ", [i(X), i(S1)]), !IO)),
        A, 0, S2, !IO),
    io.nl(!IO),
    io.write_line(S2, !IO),
    array.foldl3(three, A, 0, X3, 0, Y3, "", Z3),
    io.write_line({X3, Y3, Z3}, !IO),
    array.foldl4(four, A, 0, W4, 0, X4, 0, Y4, 0, Z4),
    io.write_line({W4, X4, Y4, Z4}, !IO),
    array.foldl5(five, A, 0, V5, 0, W5, 0, X5, 0, Y5, !IO),
    io.nl(!IO),
    io.write_line({V5, W5, X5, Y5}, !IO),
    ( if array.foldl3(three_small, A, 0, _, 0, _, "", Z6) then
        io.write_line(Z6, !IO)
    else
        io.write_string("too big\n", !IO)
    ).

:- pred sum_product(int::in, int::in, int::out, int::in, int::out) is det.
sum_product(X, S0, S0 + X, P0, P0 * X).

:- pred three(int::in, int::in, int::out, int::in, int::out,
    string::in, string::out) is det.
three(X, A0, A0 + X, B0, B0 - X, C0, C0 ++ string.int_to_string(X)).

:- pred three_small(int::in, int::in, int::out, int::in, int::out,
    string::in, string::out) is semidet.
three_small(X, A0, A0 + X, B0, B0 - X, C0, C0 ++ string.int_to_string(X)) :-
    X < 4.

:- pred four(int::in, int::in, int::out, int::in, int::out,
    int::in, int::out, int::in, int::out) is det.
four(X, A0, A0 + X, B0, B0 + 2 * X, C0, C0 + 3 * X, D0, D0 + 4 * X).

:- pred five(int::in, int::in, int::out, int::in, int::out,
    int::in, int::out, int::in, int::out, io::di, io::uo) is det.
five(X, A0, A0 + X, B0, B0 + X * X, C0, max(C0, X), D0, D0 - X, !IO) :-
    io.write_int(X, !IO).
