% A list built front to back through a hole bound at every step, then
% summed by reading it back as an open term: E5's shape (design document,
% "E5. Open-term representations"), compiled (README.md).
%
%   open-holes N R
%
% builds and sums [1, ..., N] R times and prints the total.

:- module open_holes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, string.

    % Binds L to [I | T], T a new hole, until I > N, which closes it.
:- pred build(int::in, int::in, open(list(int))::ia) is semidet.
build(I, N, L) :-
    ( if I > N then
        L = []
    else
        L = [I | T],
        build(I + 1, N, T)
    ).

:- pred sum(open(list(int))::ia, int::in, int::out) is semidet.
sum(L, A0, A) :-
    promise_pure ( if L = [X | T] then
        sum(T, A0 + X, A)
    else
        A = A0
    ).

:- pred rounds(int::in, int::in, int::in, int::out) is det.
rounds(R, N, A0, A) :-
    ( if R = 0 then
        A = A0
    else
        promise_pure ( if build(1, N, L), sum(L, 0, S) then
            A1 = A0 + S
        else
            A1 = A0
        ),
        rounds(R - 1, N, A1, A)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if
        Args = [NText, RText],
        string.to_int(NText, N),
        string.to_int(RText, R)
    then
        rounds(R, N, 0, A),
        io.format("%d elements %d rounds sum %d\n", [i(N), i(R), i(A)], !IO)
    else
        io.write_string("usage: open-holes N R\n", !IO),
        io.set_exit_status(1, !IO)
    ).
