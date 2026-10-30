% The interface section sees only the module's exported declarations and
% the modules imported in the interface (reference manual, "The module
% system"); mmc reports the same two cases.
%
% RUN: not merkc -fc1 -fdebug-dump-resolved-names %s 2>&1 | FileCheck %s

:- module interface.
:- interface.
:- pred p(list(int)::in) is det.
% CHECK: interface.m:[[@LINE-1]]:11: error: undefined type list/1
% CHECK: because: the module list has not been imported in the interface section
:- pred q(foo::in) is det.
% CHECK: interface.m:[[@LINE-1]]:11: error: undefined type foo/0
% CHECK: because: foo/0 is not exported, so it may not be used in the interface section
:- pred r(list.list(int)::in) is det.
% CHECK: interface.m:[[@LINE-1]]:11: error: undefined type list.list/1
% CHECK: because: the module list has not been imported in the interface section
:- implementation.
:- import_module list.
:- type foo ---> foo.
p(_).
q(_).
r(_).
