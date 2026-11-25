% DESIGN: Open terms (Prolog-style variables) / Open terms as built
% Open types are behind -fopen-types: the library module open is an error
% to import without it, so a program without the flag is typed as before.
% ORACLE: none
%
% RUN: not merkc -fc1 -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_without_flag.
:- interface.
:- import_module list, open.
% CHECK: reject-without-flag.m:[[@LINE-1]]:{{[0-9]+}}: error: the module open, of open types, needs -fopen-types
:- pred make(list(int)::out) is det.
:- implementation.

make(L) :-
    L = [1].
