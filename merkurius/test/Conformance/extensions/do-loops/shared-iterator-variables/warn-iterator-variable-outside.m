% DESIGN: Loops / Variables
% An iterator's own variable is local to its loop; one that also occurs
% around it is a different variable, with a warning, as the head variables
% of a lambda expression are.
% ORACLE: none
%
% RUN: merkc -fdo-loops %s -o %t 2>&1 | FileCheck %s --implicit-check-not=warning:
% RUN: %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
:- module warn_iterator_variable_outside.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    X = 7,
    ( foreach(X, [1, 2]), foreach(Y, Ys) do Y = X * 10 ),
% CHECK: warn-iterator-variable-outside.m:[[@LINE-1]]:{{[0-9]+}}: warning: variable X of the loop's iterators also occurs outside the loop, where it is a different variable
    io.write({X, Ys}, !IO), io.nl(!IO).
% OUTPUT:{7, [10, 20]}
