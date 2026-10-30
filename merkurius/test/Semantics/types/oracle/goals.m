% Each kind of goal: conjunction, disjunction, if-then-else in goals and
% in expressions, negation, quantifiers, unification of variables and
% terms, state variables, DCG rules, explicit type qualification, and
% field access and update.
%
% REQUIRES: mmc
% RUN: %compare-types %s

:- module goals.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module char, int, list, string.

:- type point ---> point(x :: int, y :: int).
:- type record ---> record(name :: string, point :: point).

main(0).
conj(X, Y) :- X = 1, Y = X + 1.
disj(X) :- ( X = 1 ; X = 2 ).
ite(X, Y) :- ( if X > 0 then Y = "pos" else Y = "nonpos" ).
ite_expr(X) = ( if X = 0 then "zero" else "other" ).
negation(X) :- not X = "a".
negation2(X) :- \+ X = 'c'.
some_goal(Xs) :- some [X] ( list.member(X, Xs), X = 1 ).
all_goal(Xs) :- all [X] ( list.member(X, Xs) => X > 0 ).
implication(X, Y) :- ( X = 1 => Y = 2 ).
equivalence(X, Y) :- ( X = 1 <=> Y = 2 ).
var_var(X, Y) :- X = Y, Y = [1].
state_vars(!S) :- !:S = !.S + 1, !:S = !.S * 2.
state_list(!L) :- !:L = [1 | !.L].
greeting --> ['h'], ['i'].
digits([D | Ds]) --> digit(D), digits(Ds).
digits([]) --> [].
digit(D) --> [D], { char.is_digit(D) }.
qualified(X) :- X = 1 : int.
qualified_list(Xs) :- Xs = [] : list(string).
get_x(P) = P ^ x.
set_x(P, X) = P ^ x := X.
deep_get(R) = R ^ point ^ y.
deep_set(R, Y) = R ^ point ^ y := Y.
field_state(!R) :- !R ^ name := "new".
switch_like(X, Y) :- ( X = 1, Y = "one" ; X = 2, Y = "two" ; Y = "many" ).
nested_ite(X) = Y :- ( if X < 0 then Y = -1 else if X > 0 then Y = 1 else Y = 0 ).
unify_tuple({A, B}, A, B).
unify_lists(Xs) :- [1, 2 | _] = Xs.
