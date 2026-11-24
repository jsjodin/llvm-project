:- module abstract_holder.
:- interface.
:- type abs(T).
:- func make(T) = abs(T).
:- implementation.
:- type abs(T)
    --->    abs(T).
make(X) = abs(X).
