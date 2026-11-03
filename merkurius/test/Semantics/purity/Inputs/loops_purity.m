% The translation of ../loops.m's loops into recursive predicates, written
% by hand: a loop's predicate is as impure as its body, and its call is
% annotated.

:- module loops_purity.
:- interface.
:- impure pred annotated(int::in) is det.
:- pred undeclared(int::in) is det.
:- pred unannotated(int::in) is det.
:- pred promised(int::in) is det.

:- implementation.
:- import_module int.

:- impure pred tick(int::in) is det.
tick(_) :-
    impure impure_true.

annotated(N) :-
    impure annotated_loop(1, N).
:- impure pred annotated_loop(int::in, int::in) is det.
annotated_loop(I, Hi) :-
    ( if I > Hi then
        true
    else
        impure tick(I),
        impure annotated_loop(I + 1, Hi)
    ).

undeclared(N) :-
    impure undeclared_loop(1, N).
:- impure pred undeclared_loop(int::in, int::in) is det.
undeclared_loop(I, Hi) :-
    ( if I > Hi then
        true
    else
        impure tick(I),
        impure undeclared_loop(I + 1, Hi)
    ).

unannotated(N) :-
    impure unannotated_loop(1, N).
:- impure pred unannotated_loop(int::in, int::in) is det.
unannotated_loop(I, Hi) :-
    ( if I > Hi then
        true
    else
        tick(I),
        impure unannotated_loop(I + 1, Hi)
    ).

promised(N) :-
    promise_pure ( impure promised_loop(1, N) ).
:- impure pred promised_loop(int::in, int::in) is det.
promised_loop(I, Hi) :-
    ( if I > Hi then
        true
    else
        impure tick(I),
        impure promised_loop(I + 1, Hi)
    ).
