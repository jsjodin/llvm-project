% Two modes of a predicate may differ only in cc-ness (reference manual,
% "Committed choice nondeterminism"). Each call runs the committed choice
% one where only its first solution is wanted, and the other elsewhere.
%
% RUN: merkc -fc1 -fdebug-dump-determinism %s | FileCheck %s
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module cc_twins.
:- interface.

:- pred digit(int).
:- mode digit(out) is multi.
:- mode digit(out) is cc_multi.

:- pred first(int::out) is cc_multi.
:- pred all(int::out) is multi.

:- implementation.

digit(X) :-
    ( X = 3 ; X = 1 ; X = 2 ).

first(X) :-
    digit(X).

all(X) :-
    digit(X).

% CHECK: 12:9 pred cc_twins.digit/1 mode 0: digit(out) is multi, inferred multi
% CHECK: 13:9 pred cc_twins.digit/1 mode 1: digit(out) is cc_multi, inferred cc_multi
% CHECK: 15:9 pred cc_twins.first/1 mode 0: first(out) is cc_multi, inferred cc_multi
% CHECK-NEXT: 24:5 call pred cc_twins.digit/1 mode 1: digit(X) [cc_multi]
% CHECK: 16:9 pred cc_twins.all/1 mode 0: all(out) is multi, inferred multi
% CHECK-NEXT: 27:5 call pred cc_twins.digit/1 mode 0: digit(X) [multi]
