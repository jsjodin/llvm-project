% The feature and inner context of run-open-types.m.
:- module nested_parallel_do_closure_open_types_lib.
:- interface.
:- import_module list, string.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int, open.

    % Built as a difference list, whose hole is closed with a ground tail.
make(N) = L :-
    make_open(1 .. N, Front, Hole),
    promise_pure (
        if close(Front, Hole, [N * 10], L0) then
            L = L0
        else
            L = []
    ).

:- pred make_open(list(int)::in,
    open(list(int))::oa, open(list(int))::oa) is det.

make_open([], T, T).
make_open([X | Xs], [X | R], T) :-
    make_open(Xs, R, T).

:- pred close(open(list(int))::ia, open(list(int))::ia,
    list(int)::in, list(int)::out) is semidet.
:- pragma promise_pure(close/4).

close(Front, Hole, Tail, List) :-
    Hole = to_open(Tail),
    impure from_open(Front, List).

    % Taken apart by unifying it with an open pattern, whose holes the
    % unification binds; then the hole of a second pattern is bound to the
    % first's rest, and closed.
use(L) = S :-
    ( if take_apart(L, A, B, Rest) then
        S = string.int_to_string(A) ++ " " ++ string.int_to_string(B) ++
            " rest " ++ string.join_list(",",
                list.map(string.int_to_string, Rest))
    else
        S = "short"
    ).

:- pred take_apart(list(int)::in, int::out, int::out, list(int)::out)
    is semidet.
:- pragma promise_pure(take_apart/4).

take_apart(L, A, B, Rest) :-
    P = [OA, OB | ORest],
    P = to_open(L),
    impure from_open(OA, A),
    impure from_open(OB, B),
    impure from_open(ORest, Rest).

    % A closure that makes and uses a value, and one that captures a value
    % made outside it.

inner_run(N) = S :-
    Make = (func(K) = make(K)),
    V = apply(Make, N),
    Use = ((func) = use(V)),
    Both = list.map((func(K) = use(make(K))), [N, N + 1]),
    S = apply(Use) ++ " | " ++ string.join_list(" | ", Both).
