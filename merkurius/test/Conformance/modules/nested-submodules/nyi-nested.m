% MANUAL: Nested submodules
% Nested submodules: one declared in the parent's interface and defined in
% its implementation, one declared and defined at once in the
% implementation, and one with only an interface, whose implicit empty
% implementation is enough as it declares only a type. A submodule sees
% the parent's implementation, and names in end_module need not be fully
% qualified.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: nested submodules

:- module nyi_nested.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

    :- module shapes.
    :- interface.
    :- type shape
        --->    square(int)
        ;       circle(int).
    :- func area(shape) = int.
    :- end_module shapes.

    :- module colours.
    :- interface.
    :- type colour
        --->    red
        ;       green.
    :- end_module colours.

:- implementation.
:- import_module int, list.
:- import_module nyi_nested.shapes, nyi_nested.colours, nyi_nested.helpers.

:- func scale = int.
scale = 3.

    :- module shapes.
    :- implementation.
    area(square(S)) = S * S.
    area(circle(R)) = scale * R * R.
    :- end_module shapes.

    :- module helpers.
    :- interface.
    :- func total(list(shape)) = int.
    :- implementation.
    total(Shapes) = list.foldl((func(S, A) = A + area(S)), Shapes, 0).
    :- end_module nyi_nested.helpers.

main(!IO) :-
    io.write_line(total([square(2), circle(1)]), !IO),
    io.write_line([red, green], !IO).
