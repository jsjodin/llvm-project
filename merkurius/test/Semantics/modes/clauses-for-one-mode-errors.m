% A clause for a mode its predicate does not have is an error, and is for
% every mode, as in mmc, which then infers wrong(in, out) multi. An
% instance's clauses cannot have modes, as in mmc (see the register,
% "Clauses for one mode in an instance"); the clause is for every mode.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module clauses_for_one_mode_errors.
:- interface.

:- pred wrong(int, int).
:- mode wrong(in, out) is det.
:- mode wrong(out, in) is semidet.

:- typeclass step(T) where [
    pred step(T::in, T::out) is det
].
:- instance step(int).

:- implementation.
:- import_module int.

% mmc: "mode annotation specifies undeclared mode `wrong(out, out)'."
:- pragma promise_equivalent_clauses(wrong/2).
wrong(X::in, Y::out) :- Y = X.
wrong(X::out, Y::out) :- X = 1, Y = 2.
% CHECK: clauses-for-one-mode-errors.m:[[@LINE-1]]:1: error: this clause is for a mode pred clauses_for_one_mode_errors.wrong/2 does not have

% mmc: "in argument 1 of clause head: error: undefined symbol `::'/2."
:- instance step(int) where [
    (step(X::in, Y::out) :- Y = X + 1)
% CHECK: clauses-for-one-mode-errors.m:[[@LINE-1]]:11: error: the clauses of an instance's methods cannot have modes
].
