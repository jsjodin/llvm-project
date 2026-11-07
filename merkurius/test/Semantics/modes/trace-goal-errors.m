% Mode errors in trace goals (reference manual, "Trace goals"), as mmc
% reports them: a trace goal may not bind a variable outside it, since it
% may not be compiled in or not run, and must give back the unique io
% state that io(!IO) gave it, without clobbering it first. The state
% variable of io(!IO) is a new one, which warns if it shadows the clause's.
% (A goal outside that binds the variable runs first, as for a negation:
% ../determinism/trace-goals.m.)
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module trace_goal_errors.
:- interface.
:- import_module io.

:- pred binds(int::out) is det.
:- pred clobbers(int::in) is det.
:- pred shares(int::in) is det.
:- pred shadows(int::in, io::di, io::uo) is det.

:- implementation.
:- import_module int.

binds(Y) :-
    trace [io(!IO)] ( Y = 3, io.write_int(Y, !IO) ).
% CHECK: trace-goal-errors.m:[[@LINE-1]]:27: error: in the clause for binds(out): scope error: Y is bound inside a trace goal, but is not local to it: it has inst free, and would get unique(3)

clobbers(X) :-
    trace [io(!IO)] ( io.write_int(X, !.IO, _) ).
% CHECK: trace-goal-errors.m:[[@LINE-1]]:23: error: in the clause for clobbers(in): unique-mode error in argument 2 of the call to pred io.write_int/3: the called procedure would clobber its argument, but IO!0 is still live

shares(X) :-
    trace [io(!IO)] ( io.write_int(X, !IO), keep(!IO) ).
% CHECK: trace-goal-errors.m:[[@LINE-1]]:5: error: in the clause for shares(in): mode error: the trace goal must give back a unique io state, but IO!2 has inst ground

:- pred keep(io::in, io::out) is det.
keep(X, X).

shadows(X, !IO) :-
    trace [io(!IO)] ( io.write_int(X, !IO) ),
% CHECK: trace-goal-errors.m:[[@LINE-1]]:15: warning: the new state variable !IO shadows the one outside
    io.write_int(X, !IO).
