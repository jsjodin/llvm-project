% A loop is as impure as its body, without an annotation of its own, as an
% if-then-else is; the calls in its body are annotated as anywhere.
% Inputs/loops_purity.m is the loops' translation, whose purity errors mmc
% reports at the same goals.
%
% RUN: not merkc -fc1 -fdo-loops -fdebug-dump-types %s 2>&1 >/dev/null \
% RUN:   | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines --allow-errors %S/Inputs/loops_purity.m %}

:- module loops.
:- interface.
:- impure pred annotated(int::in) is det.
:- pred undeclared(int::in) is det.
% CHECK: loops.m:[[@LINE-1]]:9: error: pred loops.undeclared/1 is impure, so it must be declared `impure' or promised pure
:- pred unannotated(int::in) is det.
% CHECK: loops.m:[[@LINE-1]]:9: error: pred loops.unannotated/1 is impure, so it must be declared `impure' or promised pure
:- pred promised(int::in) is det.

:- implementation.
:- import_module int.

:- impure pred tick(int::in) is det.
tick(_) :-
    impure impure_true.

annotated(N) :-
    ( for(I, 1, N) do impure tick(I) ).
undeclared(N) :-
    ( for(I, 1, N) do impure tick(I) ).
unannotated(N) :-
    ( for(I, 1, N) do tick(I) ).
% CHECK: loops.m:[[@LINE-1]]:23: error: the call of impure pred loops.tick/1 must be annotated `impure'
promised(N) :-
    promise_pure ( for(I, 1, N) do impure tick(I) ).
