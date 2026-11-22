% Unification of two open lists (README.md): one built with a hole every
% K elements, one without, unified, and the first summed, R times.

:- module open_unify.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, string.

    % Binds L to the list I, ..., N, with a new hole for every K-th element.
:- pred build(int::in, int::in, int::in, open(list(int))::ia) is semidet.
build(I, N, K, L) :-
    ( if I > N then
        L = []
    else if I mod K = 0 then
        L = [_ | T],
        build(I + 1, N, K, T)
    else
        L = [I | T],
        build(I + 1, N, K, T)
    ).

:- pred same(open(list(int))::ia, open(list(int))::ia) is semidet.
:- pragma no_inline(same/2).
same(X, Y) :- X = Y.

:- pred total(open(list(int))::ia, int::in, int::out) is semidet.
total(L, A0, A) :-
    promise_pure ( if L = [X | T] then
        total(T, A0 + X, A)
    else
        A = A0
    ).

:- pred rounds(int::in, int::in, int::in, int::in, int::out) is det.
rounds(R, N, K, A0, A) :-
    ( if R = 0 then
        A = A0
    else
        % Full is built first, without holes; the unification binds the
        % holes of Holes to Full's elements.
        promise_pure ( if
            build(1, N, N + 1, Full), build(1, N, K, Holes), same(Holes, Full),
            total(Holes, 0, S)
        then
            A1 = A0 + S
        else
            A1 = A0
        ),
        rounds(R - 1, N, K, A1, A)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if
        Args = [NText, RText],
        string.to_int(NText, N),
        string.to_int(RText, R)
    then
        rounds(R, N, 100, 0, A),
        io.format("%d elements %d rounds sum %d\n", [i(N), i(R), i(A)], !IO)
    else
        io.write_string("usage: open-unify N R\n", !IO),
        io.set_exit_status(1, !IO)
    ).
