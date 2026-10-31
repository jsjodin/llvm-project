% The instances that constraints may use are those of every interface
% loaded, also of a module that is not imported: colours's, which palette
% imports. They are found through their instances' constraints too.
%
% RUN: merkc -fc1 -fdebug-dump-types -I %S/Inputs %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --include %S/Inputs %s %}

:- module visible_instances.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module classes, list, palette.

main(0).
:- func indirect = string.
indirect = shown(favourite).
% CHECK: clause at [[@LINE-1]]:1 of func visible_instances.indirect/0
% CHECK:   constraint of shown at [[@LINE-2]]:12: classes.shown(colours.colour) by the instance at colours.m:7:13
:- func premises = string.
premises = shown([[favourite], []]).
% CHECK: clause at [[@LINE-1]]:1 of func visible_instances.premises/0
% CHECK:   constraint of shown at [[@LINE-2]]:12: classes.shown(list.list(list.list(colours.colour))) by the instance at classes.m:9:13
% CHECK-NEXT:   classes.shown(list.list(colours.colour)) by the instance at classes.m:9:13
% CHECK-NEXT:     classes.shown(colours.colour) by the instance at colours.m:7:13
