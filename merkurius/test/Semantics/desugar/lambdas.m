% Lambda expressions and conditional expressions (reference manual,
% "Lambda expressions" and "Conditional expressions"). The module compiles
% with mmc.
%
% RUN: merkc -fc1 -fdebug-dump-semantic-form %s | FileCheck %s

:- module lambdas.
:- interface.
:- import_module char, int, list.

:- pred p(int::in, int::out, int::in, int::out) is semidet.
:- func add_one(list(int)) = list(int).
:- pred digit(pred(list(char), list(char))::out(pred(in, out) is semidet))
    is det.
:- func sign(int) = int.
:- pred step(int::in, int::out) is det.

:- implementation.

:- pred q(int::in, int::out, int::in, int::out) is semidet.
q(C, C, !S) :- !:S = !.S + 1.

% The manual's example ("State variables"): the lambda's head names its
% own S, which hides the clause's.
p(A, B, !S) :-
    F =
        ( pred(C::in, D::out, !.S::in, !:S::out) is semidet :-
            q(C, D, !S)
        ),
    ( if F(A, E, !S) then
        B = E
    else
        B = A
    ).
% CHECK:      Predicate p(A, B, S!0, S!4)
% CHECK-NEXT: | Conj #10
% CHECK-NEXT: | | Unify #1 F = pred(C::in, D::out, S!1::in, S!2::out) is semidet
% CHECK-NEXT: | | | Call #0 q(C, D, S!1, S!2)
% CHECK-NEXT: | | IfThenElse #9
% CHECK-NEXT: | | | HOCall #2 F(A, E, S!0, S!3)
% CHECK-NEXT: | | | Conj #8
% CHECK-NEXT: | | | | Unify #3 B = E
% CHECK-NEXT: | | | | Unify #5 S!4 = S!3
% CHECK-NEXT: | | | Conj #7
% CHECK-NEXT: | | | | Unify #4 B = A
% CHECK-NEXT: | | | | Unify #6 S!4 = S!0

% A lambda's argument variables are its own.
add_one(Xs) = list.map(func(X) = X + 1, Xs).
% CHECK:      Function add_one(Xs) = $1
% CHECK-NEXT: | Conj #5
% CHECK-NEXT: | | Unify #4 $1 = list.map($2, Xs)
% CHECK-NEXT: | | Unify #3 $2 = func(X) = ($4)
% CHECK-NEXT: | | | Conj #2
% CHECK-NEXT: | | | | Unify #1 $4 = '+'(X, $5)
% CHECK-NEXT: | | | | Unify #0 $5 = 1

% A DCG lambda's last two arguments are its implicit ones.
digit(P) :-
    P = (pred(in, out) is semidet --> [D], { char.is_digit(D) }).
% CHECK:      Predicate digit(P)
% CHECK-NEXT: | Unify #3 P = pred(dcg!0::in, dcg!1::out) is semidet
% CHECK-NEXT: | | Conj #2
% CHECK-NEXT: | | | Unify #0 dcg!0 = [D | dcg!1]
% CHECK-NEXT: | | | Call #1 char.is_digit(D)

% As in mmc, a conditional expression is an if-then-else goal that binds
% the expression's variable in each branch.
sign(N) = (if N < 0 then -1 else if N = 0 then 0 else 1).
% CHECK:      Function sign(N) = $1
% CHECK-NEXT: | IfThenElse #8
% CHECK-NEXT: | | Conj #2
% CHECK-NEXT: | | | Unify #0 $2 = 0
% CHECK-NEXT: | | | Call #1 '<'(N, $2)
% CHECK-NEXT: | | Unify #3 $1 = -1
% CHECK-NEXT: | | IfThenElse #7
% CHECK-NEXT: | | | Unify #4 N = 0
% CHECK-NEXT: | | | Unify #5 $1 = 0
% CHECK-NEXT: | | | Unify #6 $1 = 1

% The state variables are threaded through it as through that goal.
step(!S) :-
    Y = (if !.S > 10 then 0 else 1),
    !:S = !.S + Y.
% CHECK:      Predicate step(S!0, S!1)
% CHECK-NEXT: | Conj #7
% CHECK-NEXT: | | IfThenElse #5
% CHECK-NEXT: | | | Conj #2
% CHECK-NEXT: | | | | Unify #0 $2 = 10
% CHECK-NEXT: | | | | Call #1 '>'(S!0, $2)
% CHECK-NEXT: | | | Unify #3 Y = 0
% CHECK-NEXT: | | | Unify #4 Y = 1
% CHECK-NEXT: | | Unify #6 S!1 = '+'(S!0, Y)
