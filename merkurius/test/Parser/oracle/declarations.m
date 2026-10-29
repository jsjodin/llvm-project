% Declarations, lambda expressions and field access, as they appear in modules.
% Each item is read by merkc and by Mercury's own term parser, which must
% agree.
%
% REQUIRES: mmc
% RUN: %compare-terms %s

p :- \+ \+ q.
p(!IO) :- q(!IO).
p :- q :- r.
X = f(a :- b).
X = [a :- b].
p :- ( a -> b ; c ).
p :- all [X] q(X).
p :- trace [io(!IO)] q.
p :- not q.
p :- not not q.
X = (pred(Y::in) is det :- true).
X = (func(Y) = Z :- Z = Y).
X = (func(Y) = Y + 1).
:- type t ---> a ; b.
:- type t ---> a ; b(int).
:- mode m == in.
:- pred p(int::in) is det.
:- func f(int) = int.
:- inst i == bound(a ; b).
X = Y ^ field.
X = Y ^ f1 ^ f2.
X = Y ^ field := Z.
X = Y ^ f1 ^ f2 := Z.
X = (a :- b).
p :- a & b.
X = { a :- b }.
X = [a | b :- c].
