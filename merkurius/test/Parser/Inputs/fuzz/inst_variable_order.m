% An inst variable used before one that its declaration's constraints name
% first: mode analysis made only the one used, then read past them.
:- module inst_variable_order.
:- interface.
:- pred ok(T, T).
:- (mode ok(in(A), out(I)) is det) <= I =< ground.
:- implementation.
ok(X, X).
