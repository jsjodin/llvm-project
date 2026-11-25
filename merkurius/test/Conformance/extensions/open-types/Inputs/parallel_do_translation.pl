% The translation of run-parallel-do.m into Prolog, with the loops as
% sequential recursion: their sums do not depend on the order.
:- ensure_loaded(mercury_write).

read_only(X, N, S) :-
    read_only(X, 1, N, 0, S).

read_only(_, I, N, S, S) :-
    I > N, !.
read_only(X, I, N, S0, S) :-
    ( from_open(X, L) -> sum_list(L, T), S1 is S0 + T ; S1 = S0 ),
    I1 is I + 1,
    read_only(X, I1, N, S1, S).

own_term(N, S) :-
    own_term(1, N, 0, S).

own_term(I, N, S, S) :-
    I > N, !.
own_term(I, N, S0, S) :-
    L = [A, B],
    (   A = B, A = I, from_open(L, [V, W])
    ->  S1 is S0 + V * W
    ;   S1 = S0
    ),
    I1 is I + 1,
    own_term(I1, N, S1, S).

main :-
    X = [1, 2 | _],
    read_only(X, 1000, S1),
    format("unbound hole: ~d~n", [S1]),
    (   X = [_, _ | T], T = [3]
    ->  read_only(X, 1000, S2),
        format("bound hole: ~d~n", [S2])
    ;   write('could not bind the hole\n')
    ),
    own_term(1000, S3),
    format("own terms: ~d~n", [S3]).
