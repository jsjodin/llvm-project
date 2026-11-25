% A submodule's declarations are not visible in its siblings unless they
% import it (reference manual, "Visibility rules"), or their parent does:
% the parent imports only this module, so it does not see its sibling
% first's function. The parent and the sibling are separate submodules in
% Inputs; this module is compiled alone.
%
% RUN: not merkc -fc1 -fdebug-dump-resolved-names -I %S/Inputs %s 2>&1 >/dev/null \
% RUN:   | FileCheck %s --implicit-check-not=error:
% CHECK: submodule-sibling.m:[[@LINE+8]]:{{[0-9]+}}: error: undefined symbol one/0

:- module sibling_parent.second.
:- interface.
:- func two = int.
:- implementation.
:- import_module int.

two = one + 1.
