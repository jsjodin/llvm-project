% An imported abstract type that its module defines as an equivalence is a
% type of its own while types are checked (reference manual, "Abstract
% types"): a value of the type it hides is not one of it, nor the other way
% round, as mmc reports. Only after checking is it seen through.
%
% RUN: not merkc -fc1 -fdebug-dump-types -I %S/Inputs %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s

:- module hidden_equivalences.
:- interface.
:- pred main(float::out) is det.
:- implementation.
:- import_module hidden_types.

main(F) :-
    W = weight(1.5),
    F = to_float(W),
    F = to_float(2.5).
% CHECK: [[@LINE-1]]:{{[0-9]+}}: error: {{.*}}

:- func as_weight(float) = weight.
as_weight(F) = F.
% CHECK: [[@LINE-1]]:{{[0-9]+}}: error: {{.*}}
