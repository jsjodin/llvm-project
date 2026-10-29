% Compound terms, lists, tuples, higher-order terms, module qualification and
% quoted text. Each item is read by merkc and by Mercury's own term parser,
% which must agree.
%
% REQUIRES: mmc
% RUN: %compare-terms %s

X = \ Y.
X = \ \ Y.
X = '+'(A, B).
X = '-'(A).
X = P(A).
X = (P)(A).
X = P(A)(B).
X = f(a)(b).
X = [](a).
X = {}(a).
X = '[]'.
X = '{}'.
X = '[|]'(a, []).
X = [a | b].
X = [a, b | c].
X = {a, b}.
X = {}.
X = {a}.
X = (a, b).
X = f((a, b)).
X = a.b.
X = a.b.c.
X = io.write(Y).
X = list.map(F).
X = ''(a).
X = 'f'(a).
X = 'hello world'.
X = 'it''s'.
X = 'it\'s'.
X = "a\nb".
X = "say ""hi""".
X = "é".
X = $line.
X = $file.
X = f($line).
X = 1(a).
X = 1.5(a).
X = [a](b).
X = {a}(b).
X = $line(a).
X = f(a)(b).
X = [] (a).
X = '[]'(a).
X = '{}'(a).
X = f(a) - 1.
