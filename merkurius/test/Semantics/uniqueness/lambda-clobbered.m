% A lambda expression whose argument is unique and kept for backtracking,
% `unique >> mostly_clobbered`, may not clobber it: its caller may still
% use the value on backtracking. mmc accepts it, as it checks no lambda
% head's final inst when it is clobbered (see the design document's
% register), so merkc reports it with -fmmc-uniqueness too, and this test
% is not compared with mmc. An argument that its modes do not clobber
% cannot be updated in the body (mmc reports it as shared there, at the
% same line), and one that they clobber can.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.err
% RUN: not merkc -fc1 -fdebug-dump-determinism -fmmc-uniqueness %s 2>%t.mmc >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.mmc

:- module lambda_clobbered.
:- interface.
:- type t ---> t(int).
:- pred kept(t::di, t::uo, t::out) is det.
:- pred kept_named(t::di, t::uo, t::out) is det.
:- pred clobbered(t::di, t::uo, t::out) is det.
:- pred mostly_unique(t::di, t::uo, t::out) is det.
:- implementation.
:- import_module int.
:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- inst kept == unique.
:- mode kept == (kept >> mostly_clobbered).

kept(A, B, C) :- P = (pred(X::(unique >> mostly_clobbered), Y::out) is det :- updu(2, X, Y)), P(A, C), B = t(1).
% CHECK-DAG: lambda-clobbered.m:[[@LINE-1]]:23: error: in the clause for kept(di, uo, out): mode error: argument 1, LambdaHeadVar__1, did not get sufficiently instantiated: its final inst is clobbered, but mostly_clobbered is expected
kept_named(A, B, C) :- P = (pred(X::kept, Y::out) is det :- updu(2, X, Y)), P(A, C), B = t(1).
% CHECK-DAG: lambda-clobbered.m:[[@LINE-1]]:29: error: in the clause for kept_named(di, uo, out): mode error: argument 1, LambdaHeadVar__1, did not get sufficiently instantiated: its final inst is clobbered, but mostly_clobbered is expected
clobbered(A, B, C) :- P = (pred(X::di, Y::out) is det :- updu(2, X, Y)), P(A, C), B = t(1).
mostly_unique(A, B, C) :- P = (pred(X::(unique >> mostly_unique), Y::out) is det :- updu(2, X, Y)), P(A, C), B = t(1).
% CHECK-DAG: lambda-clobbered.m:[[@LINE-1]]:85: error: in the clause for mostly_unique(di, uo, out): unique-mode error in argument 2 of the call to pred lambda_clobbered.updu/3: the called procedure would clobber its argument, but X is still live
