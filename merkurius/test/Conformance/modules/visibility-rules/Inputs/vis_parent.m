:- module vis_parent.
:- interface.
    :- module public.
    :- interface.
    :- func one = int.
    :- end_module public.
:- implementation.
    :- module public.
    :- implementation.
    one = 1.
    :- end_module public.
    :- module private.
    :- interface.
    :- func two = int.
    :- implementation.
    two = 2.
    :- end_module private.
