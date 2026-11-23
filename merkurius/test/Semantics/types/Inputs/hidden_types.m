% Abstract types that are equivalences, for hidden-equivalences.m.

:- module hidden_types.
:- interface.

:- type weight.
:- func weight(float) = weight.
:- func to_float(weight) = float.

:- implementation.

:- type weight == float.

weight(F) = F.
to_float(W) = W.
