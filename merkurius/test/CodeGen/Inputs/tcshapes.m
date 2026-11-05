:- module tcshapes.
:- interface.
:- import_module list.
:- typeclass sized(T) where [
    func size(T) = int
].
:- typeclass named(T) <= sized(T) where [
    pred label(T::in, int::out) is det
].
:- typeclass deep(T) <= named(T) where [
    func depth(T) = int
].
:- type token ---> token(int).
:- type hidden.
:- func make(int) = hidden.
:- instance sized(token).
:- instance named(token).
:- instance deep(token).
:- instance sized(hidden).
:- instance sized(int).
:- instance sized(list(T)) <= sized(T).
:- func total(list(T)) = int <= sized(T).
:- implementation.
:- import_module int.
:- type hidden ---> hidden(int, int).
make(N) = hidden(N, N + 1).
:- instance sized(token) where [ size(token(N)) = N ].
:- instance named(token) where [ label(token(N), N * 2) ].
:- instance deep(token) where [ depth(_) = 3 ].
:- instance sized(hidden) where [ size(hidden(A, B)) = A + B ].
:- instance sized(int) where [ size(N) = N ].
:- instance sized(list(T)) <= sized(T) where [ func(size/1) is total ].
total([]) = 0.
total([X | Xs]) = size(X) + total(Xs).
