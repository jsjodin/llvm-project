% Garbage made beside a live list of N ints, 16 bytes a cell: R lists of
% 100 ints made and dropped. Each collection copies the live list
% (README.md).
%
%   live N R

:- module live.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- func churn(int, int) = int.
:- pragma no_inline(churn/2).
churn(R, A) = ( if R = 0 then A else churn(R - 1, A + list.length(1 .. 100)) ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if
        Args = [NText, RText],
        string.to_int(NText, N),
        string.to_int(RText, R)
    then
        Live = 1 .. N,
        C = churn(R, 0),
        io.format("live %d x %d: %d %d\n",
            [i(N), i(R), i(C), i(list.foldl(plus, Live, 0))], !IO)
    else
        io.write_string("usage: live N R\n", !IO),
        io.set_exit_status(1, !IO)
    ).
