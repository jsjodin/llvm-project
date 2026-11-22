% Garbage made under D frames: a recursion D deep, at whose bottom R lists
% of 100 ints are made and dropped. Each collection walks every frame,
% whatever survives (README.md).
%
%   deep D R

:- module deep.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

    % Not a tail call: the addition after it keeps each frame.
:- func down(int, int) = int.
down(D, R) = ( if D = 0 then churn(R, 0) else down(D - 1, R) + 1 ).

:- func churn(int, int) = int.
:- pragma no_inline(churn/2).
churn(R, A) = ( if R = 0 then A else churn(R - 1, A + list.length(1 .. 100)) ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if
        Args = [DText, RText],
        string.to_int(DText, D),
        string.to_int(RText, R)
    then
        io.format("deep %d x %d: %d\n", [i(D), i(R), i(down(D, R))], !IO)
    else
        io.write_string("usage: deep D R\n", !IO),
        io.set_exit_status(1, !IO)
    ).
