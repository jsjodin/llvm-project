% MANUAL: Field selection
% A field label makes a selection function, which fails if the term's
% constructor is not the field's. Without a declaration its modes are
% inferred at each call, and a higher-order term of it needs a lambda
% expression; with a single declared mode, the name alone makes one.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:3
% OUTPUT-NEXT:3
% OUTPUT-NEXT:5
% OUTPUT-NEXT:circle has no width
% OUTPUT-NEXT:5
% OUTPUT-NEXT:[5, 5]
% OUTPUT-NEXT:[4, 7]
% OUTPUT-NOT:{{.}}

:- module run_selection.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- type shape
    --->    circle(radius :: int)
    ;       rect(width :: int, height :: int).

:- func shape ^ width = int.
:- mode in ^ width = out is semidet.

:- implementation.
:- import_module list.

main(!IO) :-
    C = circle(3),
    R = rect(4, 5),
    io.write_line(C ^ radius, !IO),
    io.write_line(radius(C), !IO),
    io.write_line(R ^ height, !IO),
    ( if W = C ^ width then
        io.write_line(W, !IO)
    else
        io.write_string("circle has no width\n", !IO)
    ),
    ( if H = height(R) then
        io.write_line(H, !IO)
    else
        io.write_string("no height\n", !IO)
    ),
    Heights = list.filter_map((func(S) = S ^ height is semidet), [C, R, R]),
    io.write_line(Heights, !IO),
    Widths = list.filter_map(width, [R, C, rect(7, 1)]),
    io.write_line(Widths, !IO).
