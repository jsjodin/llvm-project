% A process declaration's predicates use the types io.io and
% process.mailbox, so the module must import io and process where the
% declaration is, as for any declaration that names them.
%
% RUN: not merkc -fc1 -fprocess-declarations -fdebug-dump-resolved-names %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module import.
:- interface.
:- import_module io.

:- type msg ---> m.

:- process counter(int) receives msg.
% CHECK: import.m:[[@LINE-1]]:4: error: undefined type process.mailbox/1
% CHECK: import.m:[[@LINE-2]]:4: because: the module process has not been imported in the interface section

:- implementation.
:- import_module process.

counter(_, _, !IO).
