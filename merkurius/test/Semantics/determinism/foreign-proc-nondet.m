% A foreign_proc may not be multi or nondet (reference manual, "pragma
% foreign_proc"), declared or inferred. mmc accepts one that nothing calls,
% and aborts compiling a call of one (design document, the register).
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>&1 >/dev/null \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module foreign_proc_nondet.
:- interface.
:- pred p(int::out) is multi.
:- pred q(int::in, int::out) is nondet.
:- implementation.
:- pragma foreign_proc("C", p(X::out),
    [promise_pure, will_not_call_mercury], "X = 1;").
% CHECK: foreign-proc-nondet.m:[[@LINE-2]]:4: error: p(out) is multi, but a foreign_proc may not be multi or nondet
:- pragma foreign_proc("C", q(X::in, Y::out),
    [promise_pure, will_not_call_mercury], "Y = X;").
% CHECK: foreign-proc-nondet.m:[[@LINE-2]]:4: error: q(in, out) is nondet, but a foreign_proc may not be multi or nondet
