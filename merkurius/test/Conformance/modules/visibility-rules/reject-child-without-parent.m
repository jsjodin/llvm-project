% MANUAL: Visibility rules
% It is an error to import a module without importing all of its
% parents.
% MODULES: Inputs/vis_parent.m
%
% RUN: not merkc -I %S/Inputs %s %S/Inputs/vis_parent.m -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_child_without_parent.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module vis_parent.public.                 % ERROR

main(!IO) :-
    io.write_int(one, !IO).
