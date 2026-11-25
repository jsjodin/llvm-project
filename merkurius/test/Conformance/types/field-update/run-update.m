% MANUAL: Field update
% A field label makes an update function, 'FIELD :=', whose result is a
% copy of the term with the field replaced, and which fails if the term's
% constructor is not the field's. A declared mode makes a higher-order
% term of it.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:rect(4, 5)
% OUTPUT-NEXT:rect(10, 5)
% OUTPUT-NEXT:rect(11, 5)
% OUTPUT-NEXT:circle has no width
% OUTPUT-NEXT:rect(11, 6)
% OUTPUT-NEXT:[circle(9), circle(9)]
% OUTPUT-NOT:{{.}}

:- module run_update.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- type shape
    --->    circle(radius :: int)
    ;       rect(width :: int, height :: int).

:- func 'height :='(shape, int) = shape.
:- mode 'height :='(in, in) = out is semidet.

main(!IO) :-
    R0 = rect(4, 5),
    R1 = R0 ^ width := 10,
    io.write_line(R0, !IO),
    io.write_line(R1, !IO),
    R2 = 'width :='(R1, 11),
    io.write_line(R2, !IO),
    ( if C = circle(1) ^ width := 2 then
        io.write_line(C, !IO)
    else
        io.write_string("circle has no width\n", !IO)
    ),
    Set = 'height :=',
    ( if R3 = Set(R2, 6) then
        io.write_line(R3, !IO)
    else
        io.write_string("no height\n", !IO)
    ),
    Rs = list.filter_map((func(S) = S ^ radius := 9 is semidet),
        [circle(1), R0, circle(2)]),
    io.write_line(Rs, !IO).
