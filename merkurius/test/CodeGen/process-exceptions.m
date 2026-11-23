% Exceptions in processes (docs/INPROGRESS.md, Exceptions): a process's tries
% are its own, kept in its record, so a try whose goal blocks in a receive,
% and resumes on whichever worker, still catches its throw; a process that
% catches its own exception goes on; an uncaught one ends only its process,
% which a monitor sees as failed with the exception's text, as written in
% the report, and the program goes on and exits with status 0.
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2> %t0.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=REPORT --match-full-lines --strict-whitespace < %t0.err
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2> %t2.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=REPORT --match-full-lines --strict-whitespace < %t2.err
% RUN: env MERK_WORKERS=1 %t2 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_WORKERS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_exceptions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exception, int, list, process, string, univ.

:- type bad
    --->    bad(int, string).

    % Waits for a number, inside a try, and throws it back as bad(N, ...).
:- pred catcher(mailbox(string)::in, mailbox(int)::in, io::di, io::uo)
    is det.

catcher(Reply, Box, !IO) :-
    promise_equivalent_solutions [Result, !:IO] (
        try_io(wait_and_throw(Box), Result, !IO)
    ),
    (
        Result = succeeded(N),
        send(Reply, "succeeded " ++ int_to_string(N), !IO)
    ;
        Result = exception(Univ),
        ( if univ_to_type(Univ, bad(N, S)) then
            send(Reply, "caught bad " ++ int_to_string(N) ++ " " ++ S, !IO)
        else
            send(Reply, "caught another", !IO)
        )
    ).

:- pred wait_and_throw(mailbox(int)::in, int::out, io::di, io::uo) is det.

wait_and_throw(Box, N, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = message(N0),
        L = list.map(func(X) = X * N0, 1 .. 100),
        ( if list.foldl(int.plus, L, 0) > 0 then
            throw(bad(N0, "from a receive"))
        else
            N = N0
        )
    ;
        Got = timeout,
        N = 0
    ).

    % Throws what it receives, and catches nothing.
:- pred thrower(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.

thrower(_Reply, Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = message(N),
        throw(bad(N, "uncaught"))
    ;
        Got = timeout
    ).

:- func reason(down_info) = string.

reason(down_info(_, normal)) = "normal".
reason(down_info(_, noproc)) = "noproc".
reason(down_info(_, killed)) = "killed".
reason(down_info(_, kernel_ended)) = "kernel_ended".
reason(down_info(_, noconnection)) = "noconnection".
reason(down_info(_, failed(Text))) = "failed: " ++ Text.

main(!IO) :-
    new_mailbox(Replies, !IO),
    % OUTPUT:caught bad 1 from a receive
    % OUTPUT-NEXT:caught bad 2 from a receive
    % OUTPUT-NEXT:caught bad 3 from a receive
    list.foldl(catch_in_process(Replies), [1, 2, 3], !IO),
    new_mailbox(Downs, !IO),
    new_mailbox(Unused, !IO),
    spawn(thrower(Unused), Thrower, !IO),
    monitor(Thrower, Downs, reason, _, !IO),
    send(Thrower, 7, !IO),
    % OUTPUT-NEXT:failed: bad(7, "uncaught")
    % REPORT:Uncaught Merkurius exception:
    % REPORT-NEXT:bad(7, "uncaught")
    receive(Downs, any, infinity, Down, !IO),
    ( if Down = message(Text) then
        io.write_string(Text ++ "\n", !IO)
    else
        io.write_string("no down\n", !IO)
    ),
    % OUTPUT-NEXT:main goes on
    io.write_string("main goes on\n", !IO).

:- pred catch_in_process(mailbox(string)::in, int::in, io::di, io::uo)
    is det.

catch_in_process(Replies, N, !IO) :-
    spawn(catcher(Replies), Box, !IO),
    send(Box, N, !IO),
    receive(Replies, any, infinity, Got, !IO),
    ( if Got = message(Text) then
        io.write_string(Text ++ "\n", !IO)
    else
        io.write_string("no reply\n", !IO)
    ).
