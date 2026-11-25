% A nested submodule in a module that this one imports is not yet
% implemented either; the error, in the imported file, says how to make
% it a separate submodule. The interface before it is still read.
%
% RUN: not merkc -fc1 -fdebug-dump-resolved-names -I %S/Inputs %s 2>&1 >/dev/null \
% RUN:   | FileCheck %s --implicit-check-not=error:
% CHECK: nested_holder.m:4:15: error: not yet implemented: nested submodules; nested_holder.inner can be a separate submodule, in a file of its own, nested_holder.inner.m, which the interface section of nested_holder includes with `:- include_module inner.`

:- module nested_submodule_imported.
:- interface.
:- func value = int.
:- implementation.
:- import_module nested_holder.

value = one.
