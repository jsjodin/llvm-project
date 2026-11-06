% The standard library module pair, with the declarations of the library
% reference manual (22.01).

:- module pair.
:- interface.

:- type pair(T1, T2)
    --->    (T1 - T2).
:- type pair(T) ==  pair(T, T).
:- inst pair(I1, I2) for pair/2
    --->    (I1 - I2).
:- inst pair(I) == pair(I, I).
:- func fst(pair(X, Y)) = X.
:- pred fst(pair(X, Y)::in, X::out) is det.
:- func snd(pair(X, Y)) = Y.
:- pred snd(pair(X, Y)::in, Y::out) is det.
:- func pair(T1, T2) = pair(T1, T2).

:- implementation.

fst(X - _) = X.
fst(P, fst(P)).
snd(_ - Y) = Y.
snd(P, snd(P)).
pair(X, Y) = X - Y.
