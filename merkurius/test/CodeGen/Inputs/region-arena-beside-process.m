% Target regions that run while a GPU process waits, for
% omp-target-region-arena-gpu.test: no moment comes when nothing runs on the
% device, so each region's cells must go when it ends.
:- module region_arena_beside_process.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process.

:- func build(int, list(int)) = list(int).
build(N, Xs) = ( if N = 0 then Xs else build(N - 1, [N | Xs]) ).

:- func sum(list(int), int) = int.
sum([], S) = S.
sum([X | Xs], S) = sum(Xs, S + X).

    % Waits for a number, and sends it back.
:- process echo(mailbox(int)) receives int on gpu.

echo(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        send(Reply, N, !IO)
    else
        true
    ).

    % Each region makes about 3 MB of cells, and the teams of each loop
    % many short lists.
:- pred rounds(int::in, int::in, int::out) is det.
rounds(R, Total0, Total) :-
    ( if R > 0 then
        %$omp target
        Sum = sum(build(100000, []), 0),
        short_lists(10000, Short),
        rounds(R - 1, Total0 + Sum + Short, Total)
    else
        Total = Total0
    ).

:- pred short_lists(int::in, int::out) is det.
short_lists(N, Sum) :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 1, N), fromto(0, S0, S1, Sum) do
        S1 = S0 + sum(build(I mod 8, []), 0)
    ).

main(!IO) :-
    new_mailbox(Me, !IO),
    spawn(gpu(0), echo(Me), Echo, !IO),
    rounds(40, 0, Total),
    io.write_int(Total, !IO), io.nl(!IO),
    send(Echo, 7, !IO),
    receive(Me, any, after(10000), R, !IO),
    ( if R = message(N) then
        io.write_int(N, !IO), io.nl(!IO)
    else
        io.write_string("no answer\n", !IO)
    ).
