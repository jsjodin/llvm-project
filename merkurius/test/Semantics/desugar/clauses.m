% Desugaring into the semantic form: head arguments become distinct
% variables, nested functors become chains of unifications through fresh
% variables, and implications, all and \= become negations. The module
% compiles with mmc.
%
% RUN: merkc -fc1 -fdebug-dump-semantic-form %s | FileCheck %s

:- module clauses.
:- interface.
:- import_module list.

:- pred len(list(T)::in, int::out) is det.
:- func double(int) = int.
:- pred all_positive(list(int)::in) is semidet.
:- pred pick(T::in, T::in, T::out) is multi.
:- pred apply_twice(pred(int, int)::in(pred(in, out) is det), int::in,
    int::out) is det.

:- implementation.
:- import_module int.

len([], 0).
% CHECK:      Predicate len($0, $1)
% CHECK-NEXT: | Conj #2
% CHECK-NEXT: | | Unify #0 $0 = []
% CHECK-NEXT: | | Unify #1 $1 = 0
len([_ | Xs], N) :-
    len(Xs, N0),
    N = N0 + 1.
% CHECK-NEXT: Predicate len($0, N)
% CHECK-NEXT: | Conj #4
% CHECK-NEXT: | | Unify #0 $0 = [_$1 | Xs]
% CHECK-NEXT: | | Call #1 len(Xs, N0)
% CHECK-NEXT: | | Unify #3 N = '+'(N0, $5)
% CHECK-NEXT: | | Unify #2 $5 = 1

double(X) = X * 2.
% CHECK-NEXT: Function double(X) = $1
% CHECK-NEXT: | Conj #2
% CHECK-NEXT: | | Unify #1 $1 = '*'(X, $2)
% CHECK-NEXT: | | Unify #0 $2 = 2

% The negation of the implication's negation is the goal inside, as in
% mmc, and the inner negation's number is left unused.
all_positive(Xs) :-
    all [X] (list.member(X, Xs) => X > 0),
    Xs \= [].
% CHECK-NEXT: Predicate all_positive(Xs)
% CHECK-NEXT: | Conj #11
% CHECK-NEXT: | | Not #8
% CHECK-NEXT: | | | Scope #7 Some [X]
% CHECK-NEXT: | | | | Conj #5
% CHECK-NEXT: | | | | | Call #0 list.member(X, Xs)
% CHECK-NEXT: | | | | | Not #4
% CHECK-NEXT: | | | | | | Conj #3
% CHECK-NEXT: | | | | | | | Unify #1 $2 = 0
% CHECK-NEXT: | | | | | | | Call #2 '>'(X, $2)
% CHECK-NEXT: | | Not #10
% CHECK-NEXT: | | | Unify #9 Xs = []

pick(X, Y, Z) :-
    ( Z = X
    ; Z = Y
    ).
% CHECK-NEXT: Predicate pick(X, Y, Z)
% CHECK-NEXT: | Disj #2
% CHECK-NEXT: | | Unify #0 Z = X
% CHECK-NEXT: | | Unify #1 Z = Y

apply_twice(P, X, Z) :-
    call(P, X, Y),
    P(Y, Z).
% CHECK-NEXT: Predicate apply_twice(P, X, Z)
% CHECK-NEXT: | Conj #2
% CHECK-NEXT: | | HOCall #0 P(X, Y)
% CHECK-NEXT: | | HOCall #1 P(Y, Z)
