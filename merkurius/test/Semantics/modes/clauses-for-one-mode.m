% Clauses for one mode (reference manual, "Different clauses for different
% modes"): each procedure has the clauses for every mode and those whose
% head's modes are its own, which match by what the modes mean, as
% `ground >> ground` matches `in`; a procedure that no clause is for fails.
% The errors are mmc's; clauses-for-one-mode-errors.m has those of the
% clauses themselves.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err

:- module clauses_for_one_mode.
:- interface.
:- import_module int.

:- pred both(int, int).
:- mode both(in, out) is det.
:- mode both(out, in) is det.

:- pred also(int, int).
:- mode also(in, out) is det.
:- mode also(out, in) is det.

:- pred none(int, int).
:- mode none(in, out) is det.
:- mode none(out, in) is det.

:- func twice(int) = int.

:- implementation.

:- pragma promise_equivalent_clauses(both/2).
both(X::in, Y::out) :- Y = X + 1.
both(X::out, Y::(ground >> ground)) :- X = Y - 1.
% CHECK-LABEL: both(in, out) is det
% CHECK-NEXT: construct V_2 = 1
% CHECK-NEXT: call func int.'+'/2 mode 0: Y = +(X, V_2)
% CHECK-NEXT: both(out, in) is det
% CHECK-NEXT: construct V_2 = 1
% CHECK-NEXT: call func int.'-'/2 mode 0: X = -(Y, V_2)
% CHECK-NEXT: also(in, out) is det
% CHECK-NEXT: disjunction

% A clause for every mode is one of each procedure's, so also(in, out)
% has two and also(out, in) one.
:- pragma promise_equivalent_clauses(also/2).
also(X::in, Y::out) :- Y = X.
also(X, Y) :- X = 0, Y = 0.
% ERR: clauses-for-one-mode.m:20:9: error: the determinism declaration of also(in, out) is not satisfied: it is declared det, but inferred multi
% ERR: clauses-for-one-mode.m:[[@LINE-3]]:1: because: the disjunction has more than one disjunct with solutions
% ERR: clauses-for-one-mode.m:21:9: error: the determinism declaration of also(out, in) is not satisfied: it is declared det, but inferred semidet
% ERR: clauses-for-one-mode.m:[[@LINE-4]]:26: because: the unification of Y and 0 can fail

:- pragma promise_equivalent_clauses(none/2).
none(X::in, Y::out) :- Y = X.
% ERR: clauses-for-one-mode.m:25:9: error: the determinism declaration of none(out, in) is not satisfied: it is declared det, but inferred failure
% CHECK-LABEL: none(out, in) is det
% CHECK-NEXT: [[@LINE-3]]:1 fail [failure]

% A function's default mode.
:- pragma promise_equivalent_clauses(twice/1).
twice(X::in) = (Y::out) :- Y = X * 2.
% CHECK-LABEL: twice(in) = out is det
% CHECK-NEXT: construct V_2 = 2
