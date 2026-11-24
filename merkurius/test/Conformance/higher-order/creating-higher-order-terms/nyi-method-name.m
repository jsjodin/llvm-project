% MANUAL: Creating higher-order terms
% A type class method of one mode made a higher-order term by its name:
% the closure calls the method of the instance for its type, at a known
% type and under a constraint.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for closures of methods

:- module nyi_method_name.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- typeclass shape(T) where [
    func area(T) = int
].

:- type square ---> square(int).

:- instance shape(square) where [
    area(square(S)) = S * S
].

:- func areas(list(T)) = list(int) <= shape(T).
areas(Shapes) = list.map(area, Shapes).

main(!IO) :-
    io.write_line(list.map(area, [square(2), square(3)]), !IO),
    io.write_line(areas([square(4)]), !IO).
