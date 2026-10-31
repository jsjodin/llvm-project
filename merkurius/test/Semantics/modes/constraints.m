% Inst parameters with inconsistent constraints in one declaration
% (reference manual, "Constrained polymorphic modes"); in a list after
% `<=', as in mmc, the first constraint of a variable is kept and others
% are not reported. And predicates without mode declarations, which mmc
% reports in the same pass; with mode inference, only exported ones.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: not merkc -fc1 -fdebug-dump-modes -finfer-modes %s 2>&1 >/dev/null | FileCheck --check-prefix=INFER --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module constraints.
:- interface.
:- pred p(T::in(I =< ground), T::out(I =< any)) is det.
% CHECK: [[@LINE-1]]:9: error: the inst parameter I has inconsistent constraints in this declaration
% INFER: [[@LINE-2]]:9: error: the inst parameter I has inconsistent constraints
:- pred q(T, T).
:- (mode q(in(I), out(I)) is det) <= (I =< ground, I =< unique).
:- pred ok(T, T).
:- (mode ok(in(I), out(I)) is det) <= I =< ground.
:- pred exported_nomode(int).
% CHECK: [[@LINE-1]]:9: error: the exported pred constraints.exported_nomode/1 has no mode declaration
% INFER: [[@LINE-2]]:9: error: the exported pred constraints.exported_nomode/1 has no mode declaration
:- implementation.
:- pred local_nomode(int).
% CHECK: [[@LINE-1]]:9: error: pred constraints.local_nomode/1 has no mode declaration
p(X, X).
q(X, X).
ok(X, X).
exported_nomode(_).
local_nomode(_).
