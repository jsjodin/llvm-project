% MANUAL: Typeclass declarations
% The manual's point type class, with a pred method and its mode and a
% func method in the default mode; a class of two parameters; and a class
% without methods, which still needs an instance.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{1.5, 1.0}
% OUTPUT-NEXT:"42"
% OUTPUT-NEXT:[7, 7]
% OUTPUT-NOT:{{.}}

:- module run_declarations.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, list, string.

:- typeclass point(T) where [
        % coords(Point, X, Y):
        %       X and Y are the cartesian coordinates of Point
        pred coords(T, float, float),
        mode coords(in, out, out) is det,

        % translate(Point, X_Offset, Y_Offset) = NewPoint:
        %       NewPoint is Point translated X_Offset units in the X direction
        %       and Y_Offset units in the Y direction
        func translate(T, float, float) = T
].

:- type coordinate
    --->    coordinate(
                float,  % X coordinate
                float   % Y coordinate
            ).

:- instance point(coordinate) where [
    pred(coords/3) is coordinate_coords,
    func(translate/3) is coordinate_translate
].

:- pred coordinate_coords(coordinate, float, float).
:- mode coordinate_coords(in, out, out) is det.

coordinate_coords(coordinate(X, Y), X, Y).

:- func coordinate_translate(coordinate, float, float) = coordinate.

coordinate_translate(coordinate(X, Y), Dx, Dy) = coordinate(X + Dx, Y + Dy).

:- typeclass convert(A, B) where [
    func convert(A) = B
].

:- instance convert(int, string) where [
    func(convert/1) is string.int_to_string
].

:- typeclass foo(T) where [].

:- instance foo(int) where [].

:- pred only_foo(T::in, list(T)::out) is det <= foo(T).
only_foo(X, [X, X]).

main(!IO) :-
    P = translate(coordinate(1.0, 2.0), 0.5, -1.0),
    coords(P, X, Y),
    io.write_line({X, Y}, !IO),
    S = convert(42) : string,
    io.write_line(S, !IO),
    only_foo(7, L),
    io.write_line(L, !IO).
