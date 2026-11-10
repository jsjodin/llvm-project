% Target regions that each receive from a mailbox of their own, which then
% closes, while a GPU process waits, for omp-target-region-rings-gpu.test:
% no moment comes when nothing runs on the device, so each mailbox's ring
% must go when the last region that attached it has ended.
:- module region_rings_beside_process.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process.

    % Waits for a number, and sends it back.
:- process echo(mailbox(int)) receives int on gpu.

echo(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        send(Reply, N, !IO)
    else
        true
    ).

:- pred rounds(int::in, int::in, int::out, io::di, io::uo) is det.
rounds(R, Total0, Total, !IO) :-
    ( if R > 0 then
        new_mailbox(Box, !IO),
        send(Box, R, !IO),
        %$omp target
        (
            receive(Box, any, after(10000), Got, !IO),
            ( if Got = message(N) then V = N else V = -1000 )
        ),
        close_mailbox(Box, !IO),
        rounds(R - 1, Total0 + V, Total, !IO)
    else
        Total = Total0
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    new_mailbox(Me, !IO),
    spawn_echo_at(gpu(0), Me, Echo, !IO),
    rounds(40, 0, Total, !IO),
    io.write_int(Total, !IO), io.nl(!IO),
    ( if Args = ["leave"] then
        % main ends while the GPU process still waits.
        true
    else
        send(Echo, 7, !IO),
        receive(Me, any, after(10000), R, !IO),
        ( if R = message(N) then
            io.write_int(N, !IO), io.nl(!IO)
        else
            io.write_string("no answer\n", !IO)
        )
    ).
