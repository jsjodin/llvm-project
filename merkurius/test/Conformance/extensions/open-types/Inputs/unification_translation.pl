% The translation of run-unification.m into Prolog: = is Prolog's, with no
% occurs check, unify_oc is unify_with_occurs_check, and to_open is the
% term itself.
:- ensure_loaded(mercury_write).

same(X, X).

bind(X, N) :-
    X = N.

aliasing(L) :-
    L = [A, B, _],
    same(A, B),
    bind(B, 5).

chain(L) :-
    L = [A, B, C],
    same(A, B),
    same(C, B),
    bind(C, 7).

clash(N, L) :-
    L = [N | _],
    L = [2 | _].

structure(X, A) :-
    X = p(1, _),
    X = p(A, _).

cyclic(S1, S2) :-
    (   X = [1 | X]
    ->  (   from_open(X, L)
        ->  format(atom(S1), "closed ~w", [L])
        ;   S1 = 'cyclic, not closed'
        )
    ;   S1 = 'not unified'
    ),
    (   unify_with_occurs_check(Y, [1 | Y])
    ->  S2 = 'unify_oc unified'
    ;   S2 = 'unify_oc failed'
    ).

undone(L, S) :-
    L = [A, _],
    ( A = 1, A = 2 -> S = both ; S = neither ).

opened(A, T) :-
    [1, 2, 3] = [A | T0],
    from_open(T0, T).

main :-
    ( aliasing(L1) -> mercury_write_line(L1) ; write('no\n') ),
    ( chain(L2) -> mercury_write_line(L2) ; write('no\n') ),
    ( clash(1, L3) -> mercury_write_line(L3) ; write('clash\n') ),
    structure(X, A), mercury_write_line(X), mercury_write_line(A),
    cyclic(S1, S2), format("~w~n~w~n", [S1, S2]),
    undone(L4, S4), format("~w ", [S4]), mercury_write_line(L4),
    (   opened(B, T)
    ->  mercury_write_line(B), mercury_write_line(T)
    ;   write('no\n')
    ).
