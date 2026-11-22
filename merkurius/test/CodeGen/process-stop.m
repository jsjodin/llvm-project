% process.stop (docs/INPROGRESS.md, Failure, step 1): the process that owns
% a mailbox ends with the reason killed, which its monitors see. A blocked
% process ends at once; one busy in a loop that never ends, at its next
% check, as preemption's, with one worker too; one that stops itself, before
% its next goal. Its mailboxes close, so a later send goes nowhere and a
% later monitor says noproc, and a request to a stopped server says noproc.
% Stopping a process that has ended does nothing, and stopping the main
% process ends the program. Processes that end by themselves wait for a go
% first, so that their monitors are made before they end. mmc has no processes, so the outputs are
% written by hand: each line says what it shows.
%
% RUN: merkc -O0 %s -o %t.0 && %t.0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.2 && %t.2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_WORKERS=1 %t.2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_WORKERS=1 %t.0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% Every message is read or freed, the stopped processes' among them.
% RUN: env MERK_CHECK_MESSAGES=1 %t.2 2>&1 >/dev/null | FileCheck %s --check-prefix=MESSAGES --match-full-lines
% MESSAGES:Merkurius runtime: 0 messages are neither read nor freed
%
% RUN: not %t.2 main 2>&1 | FileCheck %s --check-prefix=MAIN --match-full-lines
% MAIN:before
% MAIN-NEXT:Merkurius runtime: the main process was stopped

:- module process_stop.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- type happening
    --->    value(int)
    ;       went(string, down_info).

:- func went_as(string, down_info) = happening.
went_as(Name, Down) = went(Name, Down).

:- pred waiter(mailbox(int)::in, io::di, io::uo) is det.
waiter(Box, !IO) :-
    receive(Box, any, infinity, _ : received(int), !IO),
    io.write_string("the waiter went on\n", !IO).

    % Runs for years, a random-number recurrence that the optimizer cannot
    % shortcut, until stop ends it.
:- pred spin(int::in, int::in, int::out) is det.
spin(K, X, M) :-
    ( if K = 0 then
        M = X
    else
        spin(K - 1, X * 6364136223846793005 + 1442695040888963407, M)
    ).

:- pred spinner(mailbox(happening)::in, mailbox(int)::in, io::di, io::uo)
    is det.
spinner(Events, Box, !IO) :-
    send(Events, value(1), !IO),
    spin(1000000000000000000, 1, M),
    io.write_int(M, !IO),
    io.write_string("the spinner went on\n", !IO),
    close_mailbox(Box, !IO).

:- pred stops_itself(mailbox(int)::in, io::di, io::uo) is det.
stops_itself(Box, !IO) :-
    receive(Box, any, infinity, _ : received(int), !IO),
    stop(Box, !IO),
    io.write_string("went on after stopping itself\n", !IO).

:- pred quick(mailbox(int)::in, io::di, io::uo) is det.
quick(Box, !IO) :-
    receive(Box, any, infinity, _ : received(int), !IO).

:- type request
    --->    double(int, mailbox(int)).

:- pred server(mailbox(request)::in, io::di, io::uo) is det.
server(Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = message(double(N, ReplyTo)),
        send(ReplyTo, 2 * N, !IO),
        server(Box, !IO)
    ;
        Got = timeout
    ).

:- pred show(mailbox(happening)::in, io::di, io::uo) is det.
show(Events, !IO) :-
    receive(Events, any, after(10000), Got, !IO),
    (
        Got = message(value(N)),
        io.format("value %d\n", [i(N)], !IO)
    ;
        Got = message(went(Name, Down)),
        io.write_string(Name ++ ": " ++ reason_text(reason(Down)) ++ "\n",
            !IO)
    ;
        Got = timeout,
        io.write_string("nothing\n", !IO)
    ).

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if Arguments = ["main"] then
        new_mailbox(Mine : mailbox(int), !IO),
        io.write_string("before\n", !IO),
        stop(Mine, !IO),
        io.write_string("after\n", !IO)
    else
        everything(!IO)
    ).

:- pred everything(io::di, io::uo) is det.
everything(!IO) :-
    new_mailbox(Events, !IO),

    % A blocked process ends at once, and its monitor says killed.
    % OUTPUT:waiter: killed
    spawn(waiter, Waiter, !IO),
    monitor(Waiter, Events, went_as("waiter"), _, !IO),
    stop(Waiter, !IO),
    show(Events, !IO),

    % Its mailbox has closed: a send goes nowhere, a monitor says noproc.
    % OUTPUT:waiter again: noproc
    send(Waiter, 5, !IO),
    monitor(Waiter, Events, went_as("waiter again"), _, !IO),
    show(Events, !IO),

    % A process busy in a loop with no end ends at its next check.
    % OUTPUT:value 1
    % OUTPUT:spinner: killed
    spawn(spinner(Events), Spinner, !IO),
    monitor(Spinner, Events, went_as("spinner"), _, !IO),
    show(Events, !IO),
    stop(Spinner, !IO),
    show(Events, !IO),

    % A process that stops itself goes no further.
    % OUTPUT:itself: killed
    spawn(stops_itself, Itself, !IO),
    monitor(Itself, Events, went_as("itself"), _, !IO),
    send(Itself, 0, !IO),
    show(Events, !IO),

    % Stopping a process that has ended does nothing.
    % OUTPUT:quick: normal
    % OUTPUT:nothing
    spawn(quick, Quick, !IO),
    monitor(Quick, Events, went_as("quick"), _, !IO),
    send(Quick, 0, !IO),
    show(Events, !IO),
    stop(Quick, !IO),
    receive(Events, any, after(50), Late, !IO),
    ( if Late = timeout then
        io.write_string("nothing\n", !IO)
    else
        io.write_string("something\n", !IO)
    ),

    % A request to a stopped server says it went down: noproc, as it had
    % ended before the request's monitor was made.
    % OUTPUT:reply 42
    % OUTPUT:server: killed
    % OUTPUT:down noproc
    spawn(server, Server, !IO),
    request(Server, (func(R) = double(21, R)), after(10000), First, !IO),
    show_reply(First, !IO),
    monitor(Server, Events, went_as("server"), _, !IO),
    stop(Server, !IO),
    show(Events, !IO),
    request(Server, (func(R) = double(1, R)), after(10000), Second, !IO),
    show_reply(Second, !IO),

    close_mailbox(Events, !IO).

:- pred show_reply(reply(int)::in, io::di, io::uo) is det.
show_reply(reply(N), !IO) :-
    io.format("reply %d\n", [i(N)], !IO).
show_reply(timeout, !IO) :-
    io.write_string("timeout\n", !IO).
show_reply(down(Reason), !IO) :-
    io.write_string("down " ++ reason_text(Reason) ++ "\n", !IO).
