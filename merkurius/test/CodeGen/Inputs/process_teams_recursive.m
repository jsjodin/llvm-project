% GPU process kinds whose code recurses, other than by tail calls, which
% gives their module's kernel a stack that is not fixed (process-teams.m,
% process-teams-gpu.test).
:- module process_teams_recursive.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process.

    % Sends Reply the Fibonacci number of the number it receives.
:- process fibber(mailbox(int)) receives int on gpu.

fibber(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then send(Reply, fib(N), !IO) else true ).

:- func fib(int) = int.

fib(N) = ( if N < 2 then N else fib(N - 1) + fib(N - 2) ).

    % Sends Reply deep(N) for the number N it receives, which recurses N
    % calls deep.
:- process deepener(mailbox(int)) receives int on gpu.

deepener(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then send(Reply, deep(N), !IO) else true ).

:- func deep(int) = int.

deep(N) = ( if N = 0 then 0 else (deep(N - 1) * 3 + 1) mod 1000 ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    new_mailbox(Me, !IO),
    ( if Args = ["deep"] then
        spawn(gpu(0), deepener(Me), D, !IO),
        send(D, 100000, !IO)
    else if Args = ["shallow"] then
        spawn(gpu(0), deepener(Me), D, !IO),
        send(D, 5, !IO)
    else
        spawn(gpu(0), fibber(Me), D, !IO),
        send(D, 20, !IO)
    ),
    receive(Me, any, after(10000), R, !IO),
    ( if R = message(M) then io.write_int(M, !IO) else io.print(R, !IO) ),
    io.nl(!IO).
