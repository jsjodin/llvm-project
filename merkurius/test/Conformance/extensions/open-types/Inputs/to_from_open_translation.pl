% The translation of run-to-from-open.m into Prolog: to_open is the term
% itself, from_open checks that it is ground and acyclic, and closed(T) is
% an ordinary term.
:- ensure_loaded(mercury_write).

round_trip(X, Y) :-
    from_open(X, Y).

show(X) :-
    (   round_trip(X, Y)
    ->  mercury_write_line(Y)
    ;   write('from_open failed\n')
    ).

try_close(L, R) :-
    ( from_open(L, B) -> R = B ; R = [] ).

fill_later(Before, After) :-
    L = ["a", "b" | T],
    try_close(L, Before),
    T = ["c"],
    from_open(L, After).

var_nonvar([A1, A2, A3, A4]) :-
    X = 1 - H,
    ( var(H) -> A1 = "var" ; A1 = "not var" ),
    ( nonvar(X) -> A2 = "nonvar" ; A2 = "not nonvar" ),
    ( H = 2 -> true ; true ),
    ( var(H) -> A3 = "var" ; A3 = "not var" ),
    ( nonvar(H) -> A4 = "nonvar" ; A4 = "not nonvar" ).

bind(H, S) :-
    H = closed(S).

closed_hole(P) :-
    X = 1 - H,
    bind(H, "filled"),
    from_open(X, P).

written(S) :-
    L = [1, A, 3, A, B],
    with_output_to(string(S), mercury_write({L, B})).

main :-
    show(42),
    show(-7),
    show(2.5),
    show("text"),
    show(green),
    show([1, 2, 3]),
    show(["x", "y"]),
    show({1, "one", 1.5}),
    show(3 - "three"),
    show(yes(0.25)),
    show(no),
    show([circle(1.0), box(2, 3), named("n", blue)]),
    (   fill_later(Before, After)
    ->  mercury_write_line(Before), mercury_write_line(After)
    ;   write('fill_later failed\n')
    ),
    var_nonvar(Answers),
    mercury_write_line(Answers),
    ( closed_hole(P) -> mercury_write_line(P) ; write('closed_hole failed\n') ),
    written(S),
    write(S), nl.
