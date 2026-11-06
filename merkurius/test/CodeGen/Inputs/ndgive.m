:- module ndgive.
:- interface.
:- import_module list.
:- pred elem(T::out, list(T)::in) is nondet.
:- pred small(int::out) is multi.
:- implementation.
elem(X, [H | T]) :-
    ( X = H ; elem(X, T) ).
small(1).
small(2).
small(3).
