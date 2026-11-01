% Every module of the library loads, and its names resolve, without
% errors, the sound array stub (sound/array.m) and, with -fmmc-uniqueness,
% the manual's (array.m) too.
%
% RUN: merkc -fc1 -fdebug-dump-imports %s 2>&1 | FileCheck --implicit-check-not=error %s
% RUN: merkc -fc1 -fdebug-dump-resolved-names %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=NAMES %s
% RUN: merkc -fc1 -fdebug-dump-resolved-names -fmmc-uniqueness %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=NAMES %s

:- module stubs.
:- interface.
:- import_module array, bool, builtin, char, exception, float, int, io,
    list, maybe, pair, require, string, univ.

% CHECK: imports: import_module array bool builtin char exception float int io list maybe pair require string univ

% NAMES: symbols of stubs:
