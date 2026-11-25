% A submodule's declarations are not visible in its parent unless the
% parent imports it (reference manual, "Visibility rules"). The child is a
% separate submodule in Inputs; this module is compiled alone.
%
% RUN: not merkc -fc1 -fdebug-dump-resolved-names -I %S/Inputs %s 2>&1 >/dev/null \
% RUN:   | FileCheck %s --implicit-check-not=error:
% CHECK: submodule-not-imported.m:[[@LINE+8]]:{{[0-9]+}}: error: undefined symbol peek/0

:- module unimported_parent.
:- interface.
:- include_module child.
:- func value = int.
:- implementation.

value = peek.
