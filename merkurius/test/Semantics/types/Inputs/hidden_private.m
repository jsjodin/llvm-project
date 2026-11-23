% An abstract type that is an equivalence of a type of the implementation
% section, for hidden-private.m.

:- module hidden_private.
:- interface.

:- type secret.
:- func secret(int) = secret.

:- implementation.

:- type secret == private.
:- type private
    --->    private(int).

secret(N) = private(N).
