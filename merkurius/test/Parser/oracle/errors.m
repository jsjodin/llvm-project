% Items that are syntax errors, or nearly so, and recovery after them. Each
% item is read by merkc and by Mercury's own term parser, which must agree.
%
% REQUIRES: mmc
% RUN: %compare-terms %s

X = "s"(a).
X = [a | b | c].
X = [a, ].
X = f(a, ).
X = f().
X = A `f`.
X = `f` B.
X = f(a)).
X = (a.
X = | a.
p :- q, , r.
X = 1 2.
X = f (a).
