% The translation of run-negated.m into Prolog, whose negation, condition
% and disjunction undo their bindings as the test's do. The first solution
% of the disjunction is taken with once/1.
:- ensure_loaded(mercury_write).

show(Label, X) :-
    with_output_to(string(S), mercury_write(X)),
    format("~w: ~w~n", [Label, S]).

negated_unbound(X, Answer) :-
    ( \+ X = 1 -> Answer = succeeded ; Answer = failed ).

negated_bound(X, Answer) :-
    X = 2,
    ( \+ X = 1 -> Answer = succeeded ; Answer = failed ).

condition_undone(L, Answer) :-
    L = [A, B],
    ( A = 1, B = 2, A = B -> Answer = then ; Answer = else ).

condition_kept(L, Answer) :-
    L = [A, _],
    ( A = 5 -> Answer = then ; Answer = else ).

disjunct_undone(S) :-
    once(( L = [A, B, C],
           ( A = 1, B = 1, C = 1, A = 9
           ; C = 3
           ),
           with_output_to(string(S), mercury_write(L)) )).

nested(L, Answer) :-
    L = [A, B],
    ( \+ ( A = 7, A = 8 ), B = 8 -> Answer = then ; Answer = else ).

aliased(L, Answer) :-
    L = [A, B],
    A = B,
    ( A = 4, B = 5 -> Answer = then ; Answer = else ).

main :-
    negated_unbound(X1, A1),
    format("negated_unbound ~w, ", [A1]), show("X", X1),
    negated_bound(X2, A2),
    format("negated_bound ~w, ", [A2]), show("X", X2),
    condition_undone(L3, A3),
    format("condition_undone ~w, ", [A3]), show("L", L3),
    condition_kept(L4, A4),
    format("condition_kept ~w, ", [A4]), show("L", L4),
    disjunct_undone(S5),
    format("disjunct_undone, L: ~w~n", [S5]),
    nested(L6, A6),
    format("nested ~w, ", [A6]), show("L", L6),
    aliased(L7, A7),
    format("aliased ~w, ", [A7]), show("L", L7).
