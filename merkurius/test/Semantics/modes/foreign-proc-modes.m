% A foreign_proc is for one of the procedures of its predicate or function:
% the one whose modes its arguments have, once.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module foreign_proc_modes.
:- interface.
:- pred p(int::in, int::out) is det.
:- implementation.

% mmc: "`:- pragma foreign_proc' declaration for undeclared mode of
% predicate `foreign_proc_modes.p'/2."
:- pragma foreign_proc("C",
% CHECK: foreign-proc-modes.m:[[@LINE-1]]:4: error: this foreign_proc is for a mode pred foreign_proc_modes.p/2 does not have
    p(X::out, Y::in),
    [will_not_call_mercury, promise_pure],
"X = Y;").

:- pragma foreign_proc("C",
    p(X::in, Y::out),
    [will_not_call_mercury, promise_pure],
"Y = X;").

% mmc: "duplicate `:- pragma foreign_proc' declaration for this mode of
% predicate `p'/2 in C."
:- pragma foreign_proc("C",
% CHECK: foreign-proc-modes.m:[[@LINE-1]]:4: error: pred foreign_proc_modes.p/2 has two foreign_procs for C in this mode
% CHECK: foreign-proc-modes.m:[[#@LINE-9]]:4: because: the other foreign_proc is here
    p(X::in, Y::out),
    [will_not_call_mercury, promise_pure],
"Y = X;").
