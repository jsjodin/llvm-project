% Operator priorities and associativity, prefix operators, and operators used
% as constants. Each item is read by merkc and by Mercury's own term parser,
% which must agree.
%
% REQUIRES: mmc
% RUN: %compare-terms %s

X = - X.
X = - - Y.
X = - -1.
X = f(- 1).
X = f(-).
X = f(-, a).
X = [-].
X = (-).
X = - .
X = ':'.
X = (:).
X = mod.
X = (mod).
f(+).
X = \+ a.
X = ^ Y.
X = A ^ B.
X = A ^ B ^ C.
X = !Y.
X = !.Y.
X = !:Y.
X = a = b.
X = a - b - c.
X = a ** b ** c.
X = A `max` B.
X = A `int.max` B.
X = A `F` B.
X = A `mod` B.
X = A '+' B.
X = f(int :: in).
p :- ( if a then b else c ).
p :- ( if a then b else if c then d else e ).
p :- some [X] q(X).
p :- some [X] some [Y] q(X, Y).
X = A =< B.
X = (A @< B).
X = A >> B.
X = A /\ B.
X = A rem B.
X = A div B.
X = A // B.
X = A xor B.
X = a : b.
X = a :: b.
X = a ; b.
X = (a ; b).
X = (a -> b).
X = a & b.
X = - 1 ^ 2.
X = - X ^ 2.
X = -1 ^ 2.
X = a - - 1.
X = A `f` B `g` C.
X = A `f` B + C.
