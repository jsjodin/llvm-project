% The modes and determinism of loops' predicates are inferred from their
% translation: a loop with a for ends when its trip count does, and one
% without is a switch on a list bound before it or on a fromto whose Last
% has no variables. Two lists bound before the loop, in lockstep, make it
% semidet for now: the design document's software error for lists of
% different lengths is not implemented yet. A loop with nothing to end it
% is multi.
%
% RUN: not merkc -fc1 -fdo-loops -fdebug-dump-determinism %s 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s

:- module determinism.
:- interface.
:- import_module list.
:- pred squares(int::in, list(int)::out) is det.
:- pred zip_sum(list(int)::in, list(int)::in, int::out) is semidet.
:- pred copy(list(int)::in, list(int)::out) is det.
:- pred last(list(int)::in, int::out) is semidet.
:- pred endless(list(int)::out) is det.
% CHECK: determinism.m:[[@LINE-1]]:9: error: the determinism declaration of endless(out) is not satisfied: it is declared det, but inferred multi
:- implementation.
:- import_module int.

squares(N, Ys) :-
    ( for(I, 1, N), foreach(Y, Ys) do Y = I * I ).
% CHECK: [[@LINE-1]]:7 pred determinism.squares__loop_[[@LINE-1]]_7/3 inferred mode 0: squares__loop_[[@LINE-1]]_7(in, di, uo), inferred det

zip_sum(Xs, Ys, S) :-
    ( foreach(X, Xs), foreach(Y, Ys), fromto(0, S0, S1, S) do
        S1 = S0 + X * Y
    ).
% CHECK: [[@LINE-3]]:7 pred determinism.zip_sum__loop_[[@LINE-3]]_7/4 inferred mode 0: zip_sum__loop_[[@LINE-3]]_7(in, in, di, uo), inferred semidet

copy(Xs, Ys) :-
    ( foreach(X, Xs), foreach(X, Ys) do true ).
% CHECK: [[@LINE-1]]:7 pred determinism.copy__loop_[[@LINE-1]]_7/2 inferred mode 0: copy__loop_[[@LINE-1]]_7(in, out), inferred det

last(L, Last) :-
    L = [_ | _],
    ( fromto(L, [X | Xs], Xs, []), fromto(0, _, X, Last) do true ).
% CHECK: [[@LINE-1]]:7 pred determinism.last__loop_[[@LINE-1]]_7/3 inferred mode 0: last__loop_[[@LINE-1]]_7(in, di, uo), inferred det

endless(Xs) :-
    ( foreach(X, Xs) do X = 1 ).
% CHECK: [[@LINE-1]]:7 pred determinism.endless__loop_[[@LINE-1]]_7/1 inferred mode 0: endless__loop_[[@LINE-1]]_7(uo), inferred multi
