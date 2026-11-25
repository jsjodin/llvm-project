:- module nested_holder.
:- interface.
:- func one = int.
    :- module inner.
    :- interface.
    :- func two = int.
    :- end_module inner.
:- implementation.
one = 1.
