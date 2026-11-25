% The name in a module's end_module declaration need not be fully
% qualified (reference manual, "Nested submodules"): the last components of
% the module's name end it, as mmc accepts. A name that only ends with
% the same characters, not at a component, does not.
%
% RUN: merkc -fc1 -fdebug-dump-resolved-names -I %S/Inputs %s 2>&1 >/dev/null \
% RUN:   | FileCheck %s --check-prefix=SHORT --allow-empty --implicit-check-not=error:
% RUN: sed -e 's/^:- end_module child\./:- end_module end_module_names.child./' %s \
% RUN:   > %t.full.m
% RUN: merkc -fc1 -fdebug-dump-resolved-names -I %S/Inputs %t.full.m 2>&1 >/dev/null \
% RUN:   | FileCheck %s --check-prefix=SHORT --allow-empty --implicit-check-not=error:
% RUN: sed -e 's/^:- end_module child\./:- end_module hild./' %s > %t.partial.m
% RUN: not merkc -fc1 -fdebug-dump-resolved-names -I %S/Inputs %t.partial.m 2>&1 >/dev/null \
% RUN:   | FileCheck %s --check-prefix=PARTIAL --implicit-check-not=error:
% SHORT-NOT: error:
% PARTIAL: error: this end_module declaration names hild, but it ends module end_module_names.child

:- module end_module_names.child.
:- interface.
:- func one = int.
:- implementation.
one = 1.
:- end_module child.
