% A reduction in a loop that a GPU process runs across teams, which comes
% later (process-loops-errors.m).
:- module process_loop_reduction.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

:- process summer(mailbox(int)) receives int on gpu.

summer(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        %$omp distribute parallel do reduction((+) : Sum)
        ( for(I, 1, N), fromto(0, S0, S1, Sum) do
            S1 = S0 + I
        ),
        send(Reply, Sum, !IO)
    else
        true
    ).

main(!IO) :-
    new_mailbox(Me, !IO),
    spawn(summer(Me), _, !IO).
