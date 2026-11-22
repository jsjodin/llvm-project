% Naive reverse of a list of N ints, R times: almost nothing survives a
% collection (README.md).
%
%   nrev N R

:- module nrev.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- func nrev(list(int)) = list(int).
nrev([]) = [].
nrev([X | Xs]) = app(nrev(Xs), [X]).

:- func app(list(int), list(int)) = list(int).
app([], Ys) = Ys.
app([X | Xs], Ys) = [X | app(Xs, Ys)].

:- pred rounds(int::in, list(int)::in, int::in, int::out) is det.
rounds(R, L, A0, A) :-
    ( if R = 0 then
        A = A0
    else
        Rev = nrev(L),
        A1 = A0 + list.det_head(Rev),
        rounds(R - 1, L, A1, A)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if
        Args = [NText, RText],
        string.to_int(NText, N),
        string.to_int(RText, R)
    then
        rounds(R, 1 .. N, 0, A),
        io.format("nrev %d x %d: %d\n", [i(N), i(R), i(A)], !IO)
    else
        io.write_string("usage: nrev N R\n", !IO),
        io.set_exit_status(1, !IO)
    ).
