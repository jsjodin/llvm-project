:- module box_equality.
:- interface.

:- pred any_equal(T::in, T::in) is semidet.

:- implementation.

any_equal(_, _) :-
    semidet_true.
