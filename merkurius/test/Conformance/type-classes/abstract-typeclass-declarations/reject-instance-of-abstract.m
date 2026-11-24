% MANUAL: Abstract typeclass declarations
% An instance that defines methods can only be declared where the
% non-abstract typeclass declaration is visible.
% MODULES: Inputs/shapes.m
%
% RUN: not merkc -I %S/Inputs %s %S/Inputs/shapes.m -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:

:- module reject_instance_of_abstract.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module shapes.

:- type circle ---> circle(int).

:- instance shape(circle) where [                   % ERROR
    side(circle(R)) = R
].

main(!IO) :-
    io.write_line(area(square(2)), !IO).
