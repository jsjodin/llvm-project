% A '#' line number directive sets the number of the next line
% (reference manual, 2.4 Tokens).
%
% RUN: merkc -fc1 -fdebug-unparse-tokens %s | cmp - %s
% RUN: merkc -fc1 -fdebug-dump-tokens %s | FileCheck %s

:- module line_directives.
:- interface.
:- func one = int.
:- implementation.
#100
one = 1.
% CHECK: 100:1 name one
% CHECK: 100:8 end .
