% An inst declared for one type constructor, applied to values of another
% type (reference manual, "Insts, modes, and mode definitions"): at an
% argument, or within the expansion of another inst.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module for_types.
:- interface.
:- import_module list.
:- inst iy for list/1 ---> [] ; [ground | ground].
:- inst wrapped == bound(w(iy)).
:- type w ---> w(int).
:- pred p(int::in(iy)) is det.
% CHECK: [[@LINE-1]]:9: error: the inst for_types.iy/0 is for values of type list.list/1, but in argument 1 of pred for_types.p/1 it is applied to values of type int
:- pred q(list(int)::in(iy)) is det.
:- pred r(w::in(wrapped)) is det.
% CHECK: [[@LINE-1]]:9: error: the inst for_types.iy/0 is for values of type list.list/1, but in argument 1 of pred for_types.r/1, within the inst for_types.wrapped/0, it is applied to values of type int
:- implementation.
p(_).
q(_).
r(_).
