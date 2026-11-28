% MANUAL: Typeclass declarations
% DESIGN: Sized floating point types
% The manual's point type class, with a pred method and its mode and a
% func method in the default mode; a class of two parameters; and a class
% without methods, which still needs an instance. It is a template, run for
% float and float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_declarations_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_declarations_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:{1.5, 1.0}
% OUTPUT-FLOAT-NEXT:"42"
% OUTPUT-FLOAT-NEXT:[7, 7]
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_declarations_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_declarations_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:{1.5f32, 1.0f32}
% OUTPUT-FLOAT32-NEXT:"42"
% OUTPUT-FLOAT32-NEXT:[7, 7]
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_declarations.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, list, string.

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
    P = translate(coordinate(1.0@S@, 2.0@S@), 0.5@S@, -1.0@S@),
    coords(P, X, Y),
    io.write_line({X, Y}, !IO),
    S = convert(42) : string,
    io.write_line(S, !IO),
    only_foo(7, L),
    io.write_line(L, !IO).
