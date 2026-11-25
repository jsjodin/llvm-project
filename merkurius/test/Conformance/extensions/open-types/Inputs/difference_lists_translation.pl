% The translation of run-difference-lists.m into Prolog, whose terms are
% open: to_open is the term itself, and from_open checks that it is ground
% and acyclic.
:- ensure_loaded(mercury_write).

make_open([], T, T).
make_open([X | Xs], [X | R], T) :-
    make_open(Xs, R, T).

close(Front, Hole, Tail, List) :-
    Hole = Tail,
    from_open(Front, List).

append_dl(A - B, B - C, A - C).

dl_to_list(F - [], L) :-
    from_open(F, L).

test(L) :-
    D1 = [1, 2 | H1] - H1,
    D2 = [3 | H2] - H2,
    append_dl(D1, D2, D),
    dl_to_list(D, L).

first(X) :-
    test(L),
    L = [X | _].

twice(L1, L2) :-
    make_open([1, 2, 3], Front, Hole),
    close(Front, Hole, [4, 5], L1),
    ( close(Front, Hole, [6], L) -> L2 = L ; L2 = [] ).

main :-
    make_open([1, 2, 3], Front, Hole),
    (   close(Front, Hole, [4, 5], List)
    ->  mercury_write_line(List)
    ;   write('hole already bound\n')
    ),
    ( test(L) -> mercury_write_line(L) ; write('no\n') ),
    ( first(X) -> mercury_write_line(X) ; write('no\n') ),
    (   twice(A, B)
    ->  mercury_write_line(A), mercury_write_line(B)
    ;   write('no\n')
    ).
