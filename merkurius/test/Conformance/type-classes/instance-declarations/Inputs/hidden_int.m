:- module hidden_int.
:- interface.
:- type hidden.
:- func make(int) = hidden.
:- implementation.
:- type hidden == int.
make(X) = X.
