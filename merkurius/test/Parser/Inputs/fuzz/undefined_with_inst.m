% A predicate declared with an undefined inst in `with_inst`: its procedure
% has no argument modes, but its clauses have two arguments, and
% determinism analysis read past the modes.
:- module undefined_with_inst.
:- interface.
:- type ho == (pred(int, int)).
:- pred w `with_type` ho `with_inst` ho.
:- implementation.
w(1, 2).
w(2, 3).
