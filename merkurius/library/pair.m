% A declaration-only stub of the standard library module pair, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

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
