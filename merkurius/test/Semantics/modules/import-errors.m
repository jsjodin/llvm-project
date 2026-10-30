% A module that cannot be found, and a source that is another module's.
%
% RUN: not merkc -fc1 -fdebug-dump-imports -I %S/Inputs %s 2>&1 | FileCheck %s

:- module import_errors.
:- interface.
:- import_module no_such_module.
% CHECK: import-errors.m:[[@LINE-1]]:18: error: cannot find module no_such_module: there is no no_such_module.m in {{.*}}Inputs, {{.*}}lib{{/|\\}}merkurius
:- import_module misnamed.
% CHECK: misnamed.m:1:11: error: {{.*}}misnamed.m is the source of module wrong_name, not of module misnamed
:- pred p is det.
:- implementation.
p.
