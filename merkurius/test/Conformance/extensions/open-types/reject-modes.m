% DESIGN: Open terms (Prolog-style variables) / Operations, Checking rules
% closed(T) holds a ground value, so its argument may not have an open type
% or inst any. An open value is never unique. Open types have no compare,
% and a comparison of open values matches none of compare's modes, which
% take ground values. An open value is not its type's value: conversion is
% explicit. free does not match any, so a branch that leaves a variable
% free while another binds part of it is a mode error.
% ORACLE: none
%
% RUN: not merkc -fc1 -fopen-types -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_modes.
:- interface.
:- import_module list, open.
:- pred close_any(open(int)::ia, closed(open(int))::out) is det.
% CHECK-DAG: reject-modes.m:[[@LINE-1]]:34: error: closed(T) holds a ground value, so its argument may not have an open type in it
:- pred unique(open(list(int))::uo) is det.
:- pred compared(open(int)::ia, open(int)::ia, comparison_result::out) is det.
:- pred less(open(int)::ia, open(int)::ia) is semidet.
:- pred to_ground(open(int)::ia, int::out) is det.
:- pred merge(bool::in, open(list(int))::oa) is det.
:- type bool ---> yes ; no.
:- implementation.
:- import_module int.

close_any(X, C) :-
    C = closed(X).
% CHECK-DAG: reject-modes.m:[[@LINE-1]]:9: error: in the clause for close_any(ia, out): mode error in the unification of C and closed(X): closed(T) holds a ground value, but X has inst any

unique(L) :-
    L = [_].
% CHECK-DAG: reject-modes.m:[[@LINE-1]]:9: error: in the clause for unique(uo): mode error: argument 1, L, had the wrong inst: its final inst is bound('[|]'(any, bound([]))), but unique is expected

compared(X, Y, R) :-
    compare(R, X, Y).
% CHECK-DAG: reject-modes.m:[[@LINE-1]]:5: error: in the clause for compared(ia, ia, out): mode error in the call to pred builtin.compare/3: the arguments R, X, Y have insts free, any, any, which match none of its modes

less(X, Y) :-
    X < Y.
% CHECK-DAG: reject-modes.m:[[@LINE-1]]:5: error: type error in call to </2: its arguments have types open.open(int), open.open(int)

to_ground(X, N) :-
    N = X.
% CHECK-DAG: reject-modes.m:[[@LINE-1]]:5: error: type error in unification of variable N and variable X: variable N has type int, but variable X has type open.open(int)

merge(B, X) :-
    ( B = yes, X = [to_open(1) | _]
    ; B = no
% CHECK-DAG: reject-modes.m:[[@LINE-1]]:11: error: in the clause for merge(in, oa): mode mismatch in the disjunction: X has insts in its branches that do not merge
    ).
