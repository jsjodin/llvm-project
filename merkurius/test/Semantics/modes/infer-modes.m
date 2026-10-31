% With -finfer-modes, as mmc's --infer-modes, the modes of a predicate
% with a `:- pred` declaration but no mode declaration are inferred from
% its calls too, as those of one without a declaration are: here dec/2's
% first argument is used after the call, so it is (unique >> ground).
% Without it, its missing mode declaration is an error (constraints.m).
%
% RUN: merkc -fc1 -fdebug-dump-modes -finfer-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module infer_modes.
:- interface.

:- pred p(int::in, int::out) is det.

:- implementation.
:- import_module int.

p(X, Y) :-
    inc(X, Y),
    dec(Y, _).

inc(X, Y) :-
    Y = X + 1.
% CHECK-LABEL: pred infer_modes.inc/2 inferred mode 0: inc(in, uo)

:- pred dec(int, int).
dec(X, Y) :-
    Y = X - 1.
% CHECK-LABEL: pred infer_modes.dec/2 inferred mode 0: dec((unique >> ground), uo)
