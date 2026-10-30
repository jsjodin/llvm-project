:- module helper.
:- interface.
:- import_module list.
:- use_module maybe.

:- func first_or(list(T), T) = T.
:- func try_first(list(T)) = maybe.maybe(T).

:- implementation.
:- import_module int.

first_or([], D) = D.
first_or([X | _], _) = X.
try_first([]) = maybe.no.
try_first([X | _]) = maybe.yes(X).
