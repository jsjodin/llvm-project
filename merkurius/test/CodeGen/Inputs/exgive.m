:- module exgive.
:- interface.
:- typeclass sh(T) where [ func sz(T) = int ].
:- instance sh(int).
:- some [T] pred give(int::in, T::out) is det => sh(T).
:- some [T] func any(int) = T.
:- implementation.
:- import_module int.
:- instance sh(int) where [ sz(X) = X + 1 ].
give(X, X).
any(X) = X * 2.
