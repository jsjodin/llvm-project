% MANUAL: Instance declarations
% DESIGN: Sized floating point types
% The manual's two instances of point: a call of translate/3 runs the
% method of the instance for its first argument's type. It is a template,
% run for float and float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_coordinates_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_coordinates_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:coordinate(2.0, 3.0)
% OUTPUT-FLOAT-NEXT:coloured_coordinate(2.0, 3.0, rgb(1, 2, 3))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_coordinates_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_coordinates_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:coordinate(2.0f32, 3.0f32)
% OUTPUT-FLOAT32-NEXT:coloured_coordinate(2.0f32, 3.0f32, rgb(1, 2, 3))
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_coordinates.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@.

:- typeclass point(T) where [
        % coords(Point, X, Y):
        %       X and Y are the cartesian coordinates of Point
        pred coords(T, @T@, @T@),
        mode coords(in, out, out) is det,

        % translate(Point, X_Offset, Y_Offset) = NewPoint:
        %       NewPoint is Point translated X_Offset units in the X direction
        %       and Y_Offset units in the Y direction
        func translate(T, @T@, @T@) = T
].

:- type coordinate
    --->    coordinate(
                @T@,  % X coordinate
                @T@   % Y coordinate
            ).

:- instance point(coordinate) where [
    pred(coords/3) is coordinate_coords,
    func(translate/3) is coordinate_translate
].

:- pred coordinate_coords(coordinate, @T@, @T@).
:- mode coordinate_coords(in, out, out) is det.

coordinate_coords(coordinate(X, Y), X, Y).

:- func coordinate_translate(coordinate, @T@, @T@) = coordinate.

coordinate_translate(coordinate(X, Y), Dx, Dy) = coordinate(X + Dx, Y + Dy).

:- type rgb
    --->    rgb(
                int,
                int,
                int
            ).

:- type coloured_coordinate
    --->    coloured_coordinate(
                @T@,
                @T@,
                rgb
            ).

:- instance point(coloured_coordinate) where [
    pred(coords/3) is coloured_coordinate_coords,
    func(translate/3) is coloured_coordinate_translate
].

:- pred coloured_coordinate_coords(coloured_coordinate, @T@, @T@).
:- mode coloured_coordinate_coords(in, out, out) is det.

coloured_coordinate_coords(coloured_coordinate(X, Y, _), X, Y).

:- func coloured_coordinate_translate(coloured_coordinate, @T@, @T@)
    = coloured_coordinate.

coloured_coordinate_translate(coloured_coordinate(X, Y, Colour), Dx, Dy)
    = coloured_coordinate(X + Dx, Y + Dy, Colour).

main(!IO) :-
    io.write_line(translate(coordinate(1.0@S@, 1.0@S@), 1.0@S@, 2.0@S@), !IO),
    io.write_line(translate(coloured_coordinate(1.0@S@, 1.0@S@, rgb(1, 2, 3)),
        1.0@S@, 2.0@S@), !IO).
