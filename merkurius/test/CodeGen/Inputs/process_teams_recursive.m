% A GPU process kind whose code recurses, which gives its module's kernel a
% dynamic stack (process-teams.m, process-teams-gpu.test).
:- module process_teams_recursive.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

    % Sends Reply the Fibonacci number of the number it receives.
:- process fibber(mailbox(int)) receives int on gpu.

fibber(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then send(Reply, fib(N), !IO) else true ).

:- func fib(int) = int.

fib(N) = ( if N < 2 then N else fib(N - 1) + fib(N - 2) ).

main(!IO) :-
    new_mailbox(Me, !IO),
    spawn(gpu(0), fibber(Me), F, !IO),
    send(F, 20, !IO),
    receive(Me, any, after(10000), R, !IO),
    ( if R = message(M) then io.write_int(M, !IO) else io.print(R, !IO) ),
    io.nl(!IO).
