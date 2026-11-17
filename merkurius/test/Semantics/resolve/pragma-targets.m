% Every pragma that names a predicate or function names one of the
% module's, declared or with its type inferred, as mmc requires of each
% (reference manual, "Pragmas"): a name the module does not have, or an
% arity or kind it does not have, is an error. Before 2026-11-17 only the
% purity promises were checked, and no_inline of an arity that did not
% exist was accepted silently.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module pragma_targets.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(!IO) :-
    p(1, X),
    inferred(X, Y),
    io.write_int(Y + f(Y), !IO).

:- pred p(int::in, int::out) is det.
p(X, X).
:- func f(int) = int.
f(X) = X.
inferred(X, X).

:- pragma inline(p/2).
:- pragma inline(inferred/2).
:- pragma no_inline(f/1).
:- pragma inline(p/3).
% CHECK: [[@LINE-1]]:4: error: `:- pragma inline' names p/3, which this module does not declare; its arity is 2
:- pragma no_inline(p/3).
% CHECK: [[@LINE-1]]:4: error: `:- pragma no_inline' names p/3, which this module does not declare; its arity is 2
:- pragma no_inline(func(p/2)).
% CHECK: [[@LINE-1]]:4: error: `:- pragma no_inline' names p/2, which this module does not declare
:- pragma consider_used(nowhere/0).
% CHECK: [[@LINE-1]]:4: error: `:- pragma consider_used' names nowhere/0, which this module does not declare
:- pragma no_determinism_warning(nowhere/0).
% CHECK: [[@LINE-1]]:4: error: `:- pragma no_determinism_warning' names nowhere/0, which this module does not declare
:- pragma promise_equivalent_clauses(nowhere/0).
% CHECK: [[@LINE-1]]:4: error: `:- pragma promise_equivalent_clauses' names nowhere/0, which this module does not declare
:- pragma terminates(nowhere/0).
% CHECK: [[@LINE-1]]:4: error: `:- pragma terminates' names nowhere/0, which this module does not declare
:- pragma does_not_terminate(nowhere/0).
% CHECK: [[@LINE-1]]:4: error: `:- pragma does_not_terminate' names nowhere/0, which this module does not declare
:- pragma check_termination(nowhere/0).
% CHECK: [[@LINE-1]]:4: error: `:- pragma check_termination' names nowhere/0, which this module does not declare
