% Clauses for one mode each may mean different things, so a predicate or
% function with more than one mode that has them is impure, unless it is
% promised pure or to have equivalent clauses (reference manual,
% "Different clauses for different modes"); the promise of equivalence
% does not promise the clauses' own goals pure. With one mode, mode
% annotations change nothing, as in mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module mode_clauses.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module list.

main(0).
:- impure pred ip(int::in) is det.
:- pragma foreign_proc("C", ip(X::in), [will_not_call_mercury], "(void) X;").

:- pred plain(list(int), list(int)).
% CHECK: [[@LINE-1]]:9: error: pred mode_clauses.plain/2 is impure, so it must be declared `impure' or promised pure
:- mode plain(in, out) is det.
:- mode plain(out, in) is det.
plain(X::in, Y::out) :- Y = X.
plain(X::out, Y::in) :- X = Y.
:- semipure pred semipure(list(int), list(int)).
% CHECK: [[@LINE-1]]:18: error: pred mode_clauses.semipure/2 is impure, so it must be declared `impure' or promised semipure
:- mode semipure(in, out) is det.
:- mode semipure(out, in) is det.
semipure(X::in, Y::out) :- Y = X.
semipure(X::out, Y::in) :- X = Y.
:- pred equivalent(list(int), list(int)).
:- mode equivalent(in, out) is det.
:- mode equivalent(out, in) is det.
:- pragma promise_equivalent_clauses(equivalent/2).
equivalent(X::in, Y::out) :- Y = X.
equivalent(X::out, Y::in) :- X = Y.
:- pred promised(list(int), list(int)).
:- mode promised(in, out) is det.
:- mode promised(out, in) is det.
:- pragma promise_pure(promised/2).
promised(X::in, Y::out) :- Y = X.
promised(X::out, Y::in) :- X = Y.
:- impure pred declared_impure(list(int), list(int)).
:- mode declared_impure(in, out) is det.
:- mode declared_impure(out, in) is det.
declared_impure(X::in, Y::out) :- Y = X.
declared_impure(X::out, Y::in) :- X = Y.
:- pred equivalent_but_impure(int, int).
% CHECK: [[@LINE-1]]:9: error: pred mode_clauses.equivalent_but_impure/2 is impure, so it must be declared `impure' or promised pure
:- mode equivalent_but_impure(in, out) is det.
:- mode equivalent_but_impure(out, in) is det.
:- pragma promise_equivalent_clauses(equivalent_but_impure/2).
equivalent_but_impure(X::in, Y::out) :- Y = X, impure ip(1).
equivalent_but_impure(X::out, Y::in) :- X = Y.

:- pred one_mode(list(int)::in, list(int)::out) is det.
one_mode(X::in, Y::out) :- Y = X.
:- pred one_mode_declaration(list(int), list(int)).
:- mode one_mode_declaration(in, out) is det.
one_mode_declaration(X::in, Y::out) :- Y = X.
:- func function(int) = int.
:- mode function(in) = out is det.
function(X::in) = (Y::out) :- Y = X.
