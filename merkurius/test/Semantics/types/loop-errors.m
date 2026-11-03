% Type errors in loops' iterators and threaded state variables, reported
% at the iterator or the loop, and in their bodies, as anywhere.
%
% RUN: not merkc -fc1 -fdo-loops -fdebug-dump-types %s 2>&1 >/dev/null \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module loop_errors.
:- interface.
:- import_module list.

:- pred bound(float::in, int::out) is det.
:- pred not_a_list(int::in, int::out) is det.
:- pred accumulator(int::in, string::out) is det.
:- pred state(int::in, int::in, string::out) is det.
:- pred body(list(int)::in, string::out) is det.
:- pred element(list(int)::in, list(string)::out) is det.

:- implementation.
:- import_module int, string.

bound(N, Sum) :-
    ( for(I, 1, N), fromto(0, S0, S1, Sum) do S1 = S0 + I ).
% CHECK: loop-errors.m:[[@LINE-1]]:7: error: type error in for(I, 1, N): variable N has type float, but a counter and its bounds have type int

not_a_list(N, S) :-
    ( foreach(X, N) do X > 0 ),
    S = N.
% CHECK: loop-errors.m:[[@LINE-2]]:7: error: type error in foreach(X, N): variable N has type int, but a list of variable X's type, T_{{[0-9]+}}, is needed

accumulator(N, S) :-
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + I ).
% CHECK: loop-errors.m:[[@LINE-1]]:21: error: type error in fromto(0, S0, S1, S): variable S has type string, but variable S0 has type int, and the four arguments of fromto have one type

state(N, !.S, !:S) :-
    ( for(I, 1, N) do !:S = !.S + I ).
% CHECK: loop-errors.m:[[@LINE-1]]:7: error: type error in the loop: state variable !S has type int at the end of its last iteration, but string after it

body(Xs, S) :-
    ( foreach(X, Xs), fromto("", S0, S1, S) do S1 = S0 ++ X ).
% CHECK: loop-errors.m:[[@LINE-1]]:53: error: type error in unification of variable S1 and `S0 ++ X': variable S1 has type string, which no declaration of ++/2 fits

element(Xs, Ys) :-
    ( foreach(X, Xs), foreach(Y, Ys) do Y = X ).
% CHECK: loop-errors.m:[[@LINE-1]]:41: error: type error in unification of variable Y and variable X: variable Y has type string, but variable X has type int
