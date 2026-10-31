% Errors in mode declarations (reference manual, "Predicate and function
% mode declarations"): one without a predicate or function declaration of
% its arity, a mode declared twice (whatever its determinism), modes both
% in a predicate declaration and separately, a mode declaration in the
% other section from its predicate's, and an exported mode without a
% determinism.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module declarations.
:- interface.

:- pred p(int).
:- mode p(in).
% CHECK: [[@LINE-1]]:9: error: the exported pred declarations.p/1 has no determinism declaration in this mode declaration
:- mode p(out) is det.

:- pred dup(int).
:- mode dup(in) is det.
% CHECK: [[@LINE-1]]:9: error: pred declarations.dup/1 has the same mode declared twice
% CHECK: [[@LINE+1]]:9: because: the other declaration of the mode is here
:- mode dup(in) is semidet.

:- pred comb(int::in) is det.
:- mode comb(out) is det.
% CHECK: [[@LINE-1]]:9: error: pred declarations.comb/1 has its modes in its pred declaration, so it may not have a separate mode declaration

:- func f(int) = int.

:- pred r(int).

:- implementation.

:- mode f(in) is det.
% CHECK: [[@LINE-1]]:9: error: mode declaration for pred f/1 without a pred declaration; there is a func of that name and arity, whose mode declaration has a result, as in `:- mode f(in) = out'
:- mode nopred(in) is det.
% CHECK: [[@LINE-1]]:9: error: mode declaration for pred nopred/1 without a pred declaration
:- mode p(in, in) is det.
% CHECK: [[@LINE-1]]:9: error: mode declaration for pred p/2 without a pred declaration; there is a pred of that name with arity 1
:- mode r(in) is det.
% CHECK: [[@LINE-1]]:9: error: this mode declaration for pred declarations.r/1 is in the implementation section, but its pred declaration is in the interface

p(_).
dup(_).
comb(1).
f(X) = X.
r(_).
