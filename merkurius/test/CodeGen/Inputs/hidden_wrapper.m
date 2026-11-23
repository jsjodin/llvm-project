% A module whose interface uses hidden_equivalences' abstract types, for
% hidden-equivalences.m.

:- module hidden_wrapper.
:- interface.
:- import_module hidden_equivalences.

:- type parcel
    --->    parcel(weight, bag(string)).

:- func parcel(float, string) = parcel.
:- func total(parcel, parcel) = weight.

:- implementation.

parcel(F, S) = parcel(weight(F), bag(S)).
total(parcel(A, _), parcel(B, _)) = add(A, B).
