% Types share their parts, so a type can be exponentially larger as a tree
% than as the graph it is; checking must not take exponential time. pp's
% inferred type doubles in every pass, since pp is a closure of itself, and
% stops growing at a bound on its size, as at the limit on passes. In q,
% X40's type doubles 40 times, and the error message cuts it short.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck %s

:- module growth.
:- interface.
:- pred q is det.
:- implementation.

pp(_, _, S0, S) :- S0 = pp, S = S0.
% CHECK: growth.m:[[@LINE-1]]:1: error: type inference did not reach a fixpoint for pp/4: its type grows in every pass

q :-
    X0 = 1,
    X1 = {X0, X0},
    X2 = {X1, X1},
    X3 = {X2, X2},
    X4 = {X3, X3},
    X5 = {X4, X4},
    X6 = {X5, X5},
    X7 = {X6, X6},
    X8 = {X7, X7},
    X9 = {X8, X8},
    X10 = {X9, X9},
    X11 = {X10, X10},
    X12 = {X11, X11},
    X13 = {X12, X12},
    X14 = {X13, X13},
    X15 = {X14, X14},
    X16 = {X15, X15},
    X17 = {X16, X16},
    X18 = {X17, X17},
    X19 = {X18, X18},
    X20 = {X19, X19},
    X21 = {X20, X20},
    X22 = {X21, X21},
    X23 = {X22, X22},
    X24 = {X23, X23},
    X25 = {X24, X24},
    X26 = {X25, X25},
    X27 = {X26, X26},
    X28 = {X27, X27},
    X29 = {X28, X28},
    X30 = {X29, X29},
    X31 = {X30, X30},
    X32 = {X31, X31},
    X33 = {X32, X32},
    X34 = {X33, X33},
    X35 = {X34, X34},
    X36 = {X35, X35},
    X37 = {X36, X36},
    X38 = {X37, X37},
    X39 = {X38, X38},
    X40 = {X39, X39},
    X40 = "a".
% CHECK: growth.m:[[@LINE-1]]:11: error: type error in unification of variable X40 and `"a"': variable X40 has type <a type with more than 200 parts>, but `"a"' has type string
