:- module shapes.
:- interface.
:- typeclass shape(T).
:- type square ---> square(int).
:- instance shape(square).
:- func area(T) = int <= shape(T).
:- implementation.
:- import_module int.
:- typeclass shape(T) where [
    func side(T) = int
].
:- instance shape(square) where [
    side(square(S)) = S
].
area(X) = side(X) * side(X).
