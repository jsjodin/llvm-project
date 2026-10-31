% Type classes and instances for the modules that import them.

:- module classes.
:- interface.
:- import_module list.
:- typeclass shown(T) where [ func shown(T) = string ].
:- typeclass hidden(T).
:- instance shown(int).
:- instance shown(list(T)) <= shown(T).
:- implementation.
:- import_module string.
:- typeclass hidden(T) where [ func hide(T) = int ].
:- instance shown(int) where [ shown(X) = string.int_to_string(X) ].
:- instance shown(list(T)) <= shown(T) where [ shown(_) = "[...]" ].
