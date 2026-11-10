% Process kinds declared in a module's interface, for process-declarations.m:
% echo is polymorphic in its messages, and logger has no arguments.
:- module process_kinds.
:- interface.
:- import_module io, process.

:- type echo_msg(T)
    --->    echo(T, mailbox(T))
    ;       quit.

    % Echoes each message back, and counts them; quit ends it.
:- process echo(int) receives echo_msg(T).

:- type log_msg
    --->    line(string)
    ;       done(mailbox(int)).

    % Prints each line; done replies with the count of lines and ends.
:- process logger receives log_msg.

:- implementation.
:- import_module int, string.

echo(Seen, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(echo(X, Reply)) then
        send(Reply, X, !IO),
        echo(Seen + 1, Self, !IO)
    else
        true
    ).

logger(Self, !IO) :-
    logger_loop(0, Self, !IO).

:- pred logger_loop(int::in, mailbox(log_msg)::in, io::di, io::uo) is det.
logger_loop(N, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(line(S)) then
        io.write_string("log: " ++ S ++ "\n", !IO),
        logger_loop(N + 1, Self, !IO)
    else if R = message(done(Reply)) then
        send(Reply, N, !IO)
    else
        true
    ).
