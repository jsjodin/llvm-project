% Every module of the library loads, and its names resolve, without
% errors.
%
% RUN: merkc -fc1 -fdebug-dump-imports %s 2>&1 | FileCheck --implicit-check-not=error %s
% RUN: merkc -fc1 -fdebug-dump-resolved-names %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=NAMES %s

:- module stubs.
:- interface.
:- import_module bool, builtin, char, exception, float, int, io, list,
    maybe, pair, require, string, univ.

% CHECK: imports: import_module bool builtin char exception float int io list maybe pair require string univ

% NAMES: symbols of stubs:
