% MANUAL: Visibility rules
% A submodule declared only in its parent's implementation section may be
% imported only by its parent and the parent's other submodules.
% MODULES: Inputs/vis_parent.m Inputs/vis_parent.public.m Inputs/vis_parent.private.m
%
% RUN: not merkc -I %S/Inputs %s %S/Inputs/vis_parent.m %S/Inputs/vis_parent.public.m %S/Inputs/vis_parent.private.m -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_private_child.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module vis_parent, vis_parent.private.    % ERROR

main(!IO) :-
    io.write_int(two, !IO).
