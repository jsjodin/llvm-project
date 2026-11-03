% Loops, ( Iterators do Body ), with -fdo-loops (design document, "Loops"):
% the iterators' arguments around the loop are flattened before it, their
% patterns are unifications at the start of its body, the state variables
% that the body updates are threaded through it, and the variables of the
% body that occur nowhere around the loop are its own, so that loops one
% after another have different variables of the same name. Each goal's
% nonlocal variables are after "~". Semantics/types/loops.m types them.
%
% RUN: merkc -fc1 -fdo-loops -fdebug-dump-nonlocals %s 2>/dev/null | FileCheck %s
% RUN: merkc -fc1 -fdo-loops -fdebug-dump-nonlocals %s 2>&1 >/dev/null \
% RUN:   | FileCheck --check-prefix=WARN --implicit-check-not=warning: %s

:- module loops.
:- interface.
:- import_module list, pair.

:- pred sum_list(list(int)::in, int::in, int::out) is det.
:- pred siblings(int::in, int::out) is det.
:- pred pi(int::in, float::out) is det.
:- pred nest(int::in, int::out) is det.
:- pred read_only(list(int)::in, int::in, int::out) is det.
:- pred patterns(list(pair(int, int))::in, int::out, int::out) is det.
:- pred last(list(int)::in, int::out) is det.
:- pred widened(int::in, int::out) is det.
:- pred copied(int::in, int::out, int::out) is det.
:- pred outside(int::in, int::out) is det.
:- pred lambdas(int::in, list(int)::out) is det.
:- pred branches(int::in, int::in, int::out) is det.
:- pred uninitialized(int::out) is det.
:- pred copy(list(int)::in, list(int)::out) is det.

:- implementation.
:- import_module float, int.

% !Sum is threaded: its versions before the loop, at the start and the end
% of an iteration, and after it.
sum_list(Xs, !Sum) :-
    ( foreach(X, Xs) do !:Sum = !.Sum + X ).
% CHECK:      Predicate sum_list(Xs, Sum!0, Sum!3)
% CHECK-NEXT: | Loop #1 ~ [Sum!0, Xs, Sum!3]
% CHECK-NEXT: | | Foreach X, Xs
% CHECK-NEXT: | | state Sum!0 -> Sum!1 ... Sum!2 -> Sum!3
% CHECK-NEXT: | | Unify #0 Sum!2 = '+'(Sum!1, X) ~ [Sum!1, X, Sum!2]

% Each loop has its own I, S0 and S1; Sum occurs in the first loop's
% fromto, around the loops, so the second shares it.
siblings(N, Count) :-
    ( for(I, 1, N), fromto(0, S0, S1, Sum) do
        S1 = S0 + I
    ),
    ( for(I, 1, N), fromto(0, S0, S1, Count) do
        ( if I * I < Sum then S1 = S0 + 1 else S1 = S0 )
    ).
% CHECK:      Predicate siblings(N, Count)
% CHECK:      | | Loop #3 ~ [N, $2, $3, Sum]
% CHECK-NEXT: | | | For I, $2, N
% CHECK-NEXT: | | | Fromto $3, S0, S1, Sum
% CHECK-NEXT: | | | Unify #2 S1 = '+'(S0, I) ~ [I, S0, S1]
% CHECK:      | | Loop #14 ~ [N, Count, Sum, $8, $9]
% CHECK-NEXT: | | | For I$10, $8, N
% CHECK-NEXT: | | | Fromto $9, S0$11, S1$12, Count
% CHECK-NEXT: | | | IfThenElse #13 ~ [Sum, I$10, S0$11, S1$12]

% X and Y are local to each iteration, and Step, from before the loop, is
% shared.
pi(N, Pi) :-
    Step = 1.0 / float(N),
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        X = (float(I) + 0.5) * Step,
        Y = 4.0 / (1.0 + X * X),
        S1 = S0 + Y
    ),
    Pi = Step * Sum.
% CHECK:      Predicate pi(N, Pi)
% CHECK:      | | Loop #{{[0-9]+}} ~ [Step, $5, $6, $8, Sum]
% CHECK-NEXT: | | | For I, $5, $6
% CHECK-NEXT: | | | Fromto $8, S0, S1, Sum
% CHECK-NEXT: | | | Conj #{{[0-9]+}} ~ [Step, I, S0, S1]

% The inner loop's bounds are computed in the outer loop's body, and the
% outer loop's I occurs around the inner loop, so the inner loop shares it.
nest(N, Total) :-
    ( for(I, 1, N), fromto(0, T0, T1, Total) do
        ( for(J, 1, I), fromto(T0, U0, U1, T1) do
            U1 = U0 + I * J
        )
    ).
% CHECK:      Predicate nest(N, Total)
% CHECK:      | | Loop #{{[0-9]+}} ~ [N, Total, $2, $3]
% CHECK-NEXT: | | | For I, $2, N
% CHECK-NEXT: | | | Fromto $3, T0, T1, Total
% CHECK-NEXT: | | | Conj #{{[0-9]+}} ~ [I, T0, T1]
% CHECK-NEXT: | | | | Unify #{{[0-9]+}} $7 = 1 ~ [$7]
% CHECK-NEXT: | | | | Loop #{{[0-9]+}} ~ [I, T0, T1, $7]
% CHECK-NEXT: | | | | | For J, $7, I
% CHECK-NEXT: | | | | | Fromto T0, U0, U1, T1
% CHECK-NEXT: | | | | | Conj #{{[0-9]+}} ~ [I, J, U0, U1]

% A body that only reads !.S shares it: it is not threaded.
read_only(Xs, !S) :-
    ( foreach(X, Xs), fromto(0, C0, C1, C) do C1 = C0 + X * !.S ),
    !:S = !.S + C.
% CHECK:      Predicate read_only(Xs, S!0, S!1)
% CHECK:      | | Loop #{{[0-9]+}} ~ [S!0, Xs, $2, C]
% CHECK-NEXT: | | | Foreach X, Xs
% CHECK-NEXT: | | | Fromto $2, C0, C1, C
% CHECK-NEXT: | | | Conj

% The patterns of the iterators' own arguments are unifications at the
% start of the body, and so is an Out that is a term.
patterns(Pairs, Keys, N) :-
    ( foreach(K - V, Pairs), fromto(0, KS, KS + K, Keys), count(_, 0, N) do
        V > 0
    ).
% CHECK:      Predicate patterns(Pairs, Keys, N)
% CHECK:      | | Loop #{{[0-9]+}} ~ [Pairs, Keys, N, $3, $4]
% CHECK-NEXT: | | | Foreach $5, Pairs
% CHECK-NEXT: | | | Fromto $3, KS, $9, Keys
% CHECK-NEXT: | | | Count _$10, $4, N
% CHECK-NEXT: | | | Conj #{{[0-9]+}} ~ [$5, KS, $9]
% CHECK-NEXT: | | | | Unify #{{[0-9]+}} $5 = '-'(K, V) ~ [$5, K, V]
% CHECK-NEXT: | | | | Unify #{{[0-9]+}} $9 = '+'(KS, K) ~ [K, KS, $9]

% A fromto whose Last has no variables ends the loop. X passes the element
% from one fromto to the other, in different places, so no warning.
last(L, Last) :-
    ( fromto(L, [X | Xs], Xs, []), fromto(0, _, X, Last) do true ).
% CHECK:      Predicate last(L, Last)
% CHECK:      | | Loop #{{[0-9]+}} ~ [L, Last, $2, $3]
% CHECK-NEXT: | | | Fromto L, $4, Xs, $2 ends
% CHECK-NEXT: | | | Fromto $3, _$7, X, Last
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $4 = [X | Xs] ~ [$4, X, Xs]

% A quantifier around the loops widens X's scope, so both loops share it.
widened(N, Y) :-
    some [X] (
        X = N + 1,
        ( for(I, 1, N), fromto(0, A0, A1, A) do A1 = A0 + I * X ),
        ( for(I, 1, N), fromto(A, B0, B1, Y) do B1 = B0 + I * X )
    ).
% CHECK:      Predicate widened(N, Y)
% CHECK:      | | | Loop #{{[0-9]+}} ~ [N, X, $4, $5, A]
% CHECK:      | | | Loop #{{[0-9]+}} ~ [N, Y, X, A, $11]
% CHECK-NEXT: | | | | For I$12, $11, N

% One fromto copied from another shares its In and Out.
copied(N, Sum, Count) :-
    ( for(I, 1, N), fromto(0, S0, S1, Sum), fromto(0, S0, S1, Count) do
        S1 = S0 + I
    ).
% WARN: loops.m:[[@LINE-3]]:55: warning: the iterators fromto(0, S0, S1, Sum) and fromto(0, S0, S1, Count) share the variable S0, so they must agree in every iteration; give each iterator its own variables
% WARN: loops.m:[[@LINE-4]]:59: warning: the iterators fromto(0, S0, S1, Sum) and fromto(0, S0, S1, Count) share the variable S1, so they must agree in every iteration; give each iterator its own variables

% An iterator's variable that also occurs around the loop is another
% variable in it.
outside(I, Sum) :-
    ( for(I, 1, 3), fromto(0, S0, S1, Sum) do S1 = S0 + I ).
% WARN: loops.m:[[@LINE-1]]:11: warning: variable I of the loop's iterators also occurs outside the loop, where it is a different variable
% CHECK:      Predicate outside(I, Sum)
% CHECK:      | | Loop #{{[0-9]+}} ~ [Sum, $2, $3, $4]
% CHECK-NEXT: | | | For I$5, $2, $3

% A lambda expression in a loop's body captures the iteration's variable,
% and a loop in a lambda expression's body shares the lambda's argument.
lambdas(N, Ys) :-
    ( for(I, 1, N), foreach(Y, Ys) do
        P = (pred(Z::out) is det :- Z = I * 2),
        P(Y)
    ),
    Q = (pred(M::in, R::out) is det :-
        ( for(J, 1, M), fromto(0, R0, R1, R) do R1 = R0 + J )
    ),
    Q(N, _).
% CHECK:      Predicate lambdas(N, Ys)
% CHECK:      | | Loop #{{[0-9]+}} ~ [N, Ys, $2]
% CHECK-NEXT: | | | For I, $2, N
% CHECK-NEXT: | | | Foreach Y, Ys
% CHECK:      | | | | Loop #{{[0-9]+}} ~ [M, R, $11, $12]
% CHECK-NEXT: | | | | | For J, $11, M

% A state variable updated in branches of the body is merged there, at the
% end of each iteration.
branches(N, Limit, !:S) :-
    !:S = 0,
    ( for(I, 1, N) do
        ( if I < Limit then !:S = !.S + I else true )
    ).
% CHECK:      Predicate branches(N, Limit, S!4)
% CHECK:      | | Loop #{{[0-9]+}} ~ [N, Limit, S!0, $3, S!4]
% CHECK-NEXT: | | | For I, $3, N
% CHECK-NEXT: | | | state S!0 -> S!1 ... S!3 -> S!4
% CHECK-NEXT: | | | IfThenElse #{{[0-9]+}} ~ [Limit, S!1, I, S!3]

% A threaded state variable needs a value before the loop.
uninitialized(!:S) :-
    ( for(I, 1, 3) do !:S = I ).
% WARN: loops.m:[[@LINE-1]]:7: warning: reference to uninitialized state variable !.S

% Two foreach with one element copy a list: no warning, which is for two
% fromtos only.
copy(Xs, Ys) :-
    ( foreach(X, Xs), foreach(X, Ys) do true ).
% CHECK:      Predicate copy(Xs, Ys)
% CHECK-NEXT: | Loop #1 ~ [Xs, Ys]
% CHECK-NEXT: | | Foreach X, Xs
% CHECK-NEXT: | | Foreach X, Ys
