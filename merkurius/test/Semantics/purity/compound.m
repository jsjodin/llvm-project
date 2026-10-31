% A purity annotation is for a call or a unification, not a compound goal
% (reference manual, "Marking a goal as impure"); the goals in it are taken
% to have the annotation. mmc reports this as it reads the clause, so on
% its own.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module compound.
:- interface.
:- pred main(int::out) is det.
:- implementation.

main(0).
:- impure pred ip(int::in) is det.
:- pragma foreign_proc("C", ip(X::in), [will_not_call_mercury], "(void) X;").
:- impure pred compound is det.
compound :- impure (ip(1), ip(2)).
% CHECK: [[@LINE-1]]:13: error: `impure' annotates a call or a unification, not a compound goal
:- impure pred disjunction(int::in) is semidet.
disjunction(X) :- impure ( X = 1, ip(X) ; X = 2 ).
% CHECK: [[@LINE-1]]:19: error: `impure' annotates a call or a unification, not a compound goal
