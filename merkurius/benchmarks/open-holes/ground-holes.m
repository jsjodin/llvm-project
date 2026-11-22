% open-holes.m's list as a ground list(int), built front to back, which
% last call modulo constructor makes a loop that fills each cell's tail
% after the next is made: E5's ground front-to-back row (design document,
% "E5. Open-term representations"), compiled (README.md).
%
%   ground-holes N R
%
% builds and sums [1, ..., N] R times and prints the total.

:- module ground_holes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred build(int::in, int::in, list(int)::out) is det.
build(I, N, L) :-
    ( if I > N then
        L = []
    else
        L = [I | T],
        build(I + 1, N, T)
    ).

:- pred sum(list(int)::in, int::in, int::out) is det.
sum([], A, A).
sum([X | T], A0, A) :-
    sum(T, A0 + X, A).

:- pred rounds(int::in, int::in, int::in, int::out) is det.
rounds(R, N, A0, A) :-
    ( if R = 0 then
        A = A0
    else
        build(1, N, L),
        sum(L, 0, S),
        rounds(R - 1, N, A0 + S, A)
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
        io.write_string("usage: ground-holes N R\n", !IO),
        io.set_exit_status(1, !IO)
    ).
