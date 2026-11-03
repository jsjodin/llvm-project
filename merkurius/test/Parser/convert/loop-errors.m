% Errors in loops, read with -fdo-loops (design document, "Loops"). An item
% with an error is left out.
%
% RUN: not merkc -fc1 -fdo-loops -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module loop_errors.
:- implementation.

not_iterator(N) :-
    ( for(I, 1, N), while(I < 10) do true ).
% CHECK: loop-errors.m:[[@LINE-1]]:21: error: 'while(I < 10)' is not an iterator: expected for(I, Lo, Hi), for(I, Lo, Hi, Step), foreach(X, List), count(I, Lo, Hi) or fromto(First, In, Out, Last)

wrong_arity(Xs) :-
    ( foreach(X, Xs, _) do true ).
% CHECK: loop-errors.m:[[@LINE-1]]:7: error: 'foreach(X, Xs, _)' is not an iterator

param(Xs, Y) :-
    ( foreach(X, Xs), param(Y) do X = Y ).
% CHECK: loop-errors.m:[[@LINE-1]]:23: error: 'param(Y)': loops have no param iterator; a variable that occurs outside a loop's body is shared by its iterations

foreacharg(T) :-
    ( foreacharg(A, T) do A = 1 ).
% CHECK: loop-errors.m:[[@LINE-1]]:7: error: 'foreacharg(A, T)': loops have no foreacharg iterator, since a term's arguments may have different types

two_fors(N) :-
    ( for(I, 1, N), for(J, 1, N) do I = J ).
% CHECK: loop-errors.m:[[@LINE-1]]:21: error: a loop has at most one for iterator; count(I, Lo, Hi) counts in lockstep with it

counter(N) :-
    ( for(f(I), 1, N) do true ),
    ( count(1, 1, N) do true ).
% CHECK: loop-errors.m:[[@LINE-2]]:11: error: 'f(I)' is not a variable: the first argument of for and count is the loop's counter
% CHECK: loop-errors.m:[[@LINE-2]]:13: error: '1' is not a variable: the first argument of for and count is the loop's counter

steps(N, S) :-
    ( for(I, 1, N, S) do true ),
    ( for(I, 1, N, 0) do true ),
    ( for(I, 1, N, 2u) do true ),
    ( for(I, 1, N, 2i8) do true ),
    ( for(I, 1, N, 1 + 1) do true ).
% CHECK: loop-errors.m:[[@LINE-5]]:20: error: 'S' is not a nonzero integer literal: the step of a for iterator is one, so that its direction is known
% CHECK: loop-errors.m:[[@LINE-5]]:20: error: '0' is not a nonzero integer literal
% CHECK: loop-errors.m:[[@LINE-5]]:20: error: '2u' is not a nonzero integer literal
% CHECK: loop-errors.m:[[@LINE-5]]:20: error: '2i8' is not a nonzero integer literal
% CHECK: loop-errors.m:[[@LINE-5]]:20: error: '1 + 1' is not a nonzero integer literal

no_iterators(N) :-
    ( N do true ).
% CHECK: loop-errors.m:[[@LINE-1]]:7: error: 'N' is not an iterator

bad_body(N) :-
    ( for(_, 1, N) do 42 ).
% CHECK: loop-errors.m:[[@LINE-1]]:23: error: expected a goal, but found '42'

dcg(N) -->
    ( for(_, 1, N) do [] ).
% CHECK: loop-errors.m:[[@LINE-1]]:7: error: a loop is not a DCG goal; write it in braces, { ( ... do ... ) }, with the state threaded explicitly
