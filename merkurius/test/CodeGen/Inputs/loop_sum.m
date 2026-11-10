% A module with a do loop, for extension-modules.m.
:- module loop_sum.
:- interface.
:- import_module list.
:- func loop_sum(list(int)) = int.
:- implementation.
:- import_module int.
loop_sum(Xs) = S :-
    ( foreach(X, Xs), fromto(0, S0, S1, S) do S1 = S0 + X ).
