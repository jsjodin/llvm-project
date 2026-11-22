% Monitors (design document, "Exceptions, links and monitors"): a process's
% normal end and its failure reach a mailbox through the watcher's function
% into its message type, on the receiver's side; a process already ended gives
% the message at once; one monitor per process end; demonitor stops it; a
% message for a closed mailbox goes nowhere. The refs tell monitors apart. A
% function whose result is a constant, as reason's, has no cells to give the
% reader, and the message is still counted once.
%
% RUN: merkc -O0 %s -o %t.0 && %t.0 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.2 && %t.2 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_WORKERS=1 %t.2 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% Every message is read or freed, those that monitors made among them.
% RUN: env MERK_CHECK_MESSAGES=1 %t.2 2>&1 >/dev/null | FileCheck %s --check-prefix=MESSAGES --match-full-lines
% MESSAGES:Merkurius runtime: 0 messages are neither read nor freed

:- module process_monitor.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process, string.

:- type happening
    --->    value(int)
    ;       went(string, down_info).

:- pred stopper(mailbox(int)::in, io::di, io::uo) is det.
stopper(Box, !IO) :-
    receive(Box, any, infinity, _ : received(int), !IO).

:- pred divider(mailbox(int)::in, io::di, io::uo) is det.
divider(Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = message(N),
        io.write_int(100 // N, !IO)
    ;
        Got = timeout
    ).

:- pred show(mailbox(happening)::in, io::di, io::uo) is det.
show(Events, !IO) :-
    receive(Events, any, after(10000), Got, !IO),
    (
        Got = message(value(N)),
        show_int(N, !IO)
    ;
        Got = message(went(Name, Down)),
        io.write_string(Name ++ ": " ++ reason_text(reason(Down)) ++ "\n", !IO)
    ;
        Got = timeout,
        io.write_string("nothing\n", !IO)
    ).

:- pred show_int(int::in, io::di, io::uo) is det.
show_int(N, !IO) :-
    io.write_int(N, !IO),
    io.nl(!IO).

:- func went_as(string, down_info) = happening.
went_as(Name, Down) = went(Name, Down).

main(!IO) :-
    new_mailbox(Events, !IO),

    % An end that is normal, among other messages, which keep their order.
    spawn(stopper, Normal, !IO),
    monitor(Normal, Events, went_as("normal end"), _, !IO),
    send(Events, value(1), !IO),
    send(Normal, 0, !IO),
    show(Events, !IO),
    show(Events, !IO),

    % It has ended: the message comes at once.
    monitor(Normal, Events, went_as("already ended"), _, !IO),
    show(Events, !IO),

    % A failure, seen by two monitors.
    spawn(divider, Failing, !IO),
    monitor(Failing, Events, went_as("first"), First, !IO),
    monitor(Failing, Events, went_as("second"), Second, !IO),
    send(Failing, 0, !IO),
    show(Events, !IO),
    show(Events, !IO),
    ( if First = Second then
        io.write_string("the same ref\n", !IO)
    else
        io.write_string("different refs\n", !IO)
    ),

    % demonitor: nothing comes. A later monitor shows that it ended.
    spawn(stopper, Removed, !IO),
    monitor(Removed, Events, went_as("removed"), Ref, !IO),
    demonitor(Ref, !IO),
    demonitor(Ref, !IO),
    monitor(Removed, Events, went_as("kept"), _, !IO),
    send(Removed, 0, !IO),
    show(Events, !IO),

    % A monitor's mailbox that has closed.
    new_mailbox(Gone : mailbox(down_info), !IO),
    spawn(stopper, Quiet, !IO),
    monitor(Quiet, Gone, (func(D) = D), _, !IO),
    close_mailbox(Gone, !IO),
    send(Quiet, 0, !IO),
    send(Events, value(2), !IO),
    show(Events, !IO),

    % A message that is a constant, the reason itself, and a request to a
    % process that has ended, whose monitor's message is the reason too.
    new_mailbox(Reasons, !IO),
    spawn(stopper, Constant, !IO),
    monitor(Constant, Reasons, reason, _, !IO),
    send(Constant, 0, !IO),
    receive(Reasons, any, after(10000), GotReason, !IO),
    ( if GotReason = message(Reason) then
        io.write_string("constant: " ++ reason_text(Reason) ++ "\n", !IO)
    else
        io.write_string("nothing\n", !IO)
    ),
    request(Constant, (func(_ : mailbox(int)) = 1), after(10000), Reply, !IO),
    ( if Reply = down(Down) then
        io.write_string("request: " ++ reason_text(Down) ++ "\n", !IO)
    else
        io.write_string("a reply\n", !IO)
    ),
    close_mailbox(Reasons, !IO).

% OUTPUT:1
% OUTPUT-NEXT:normal end: normal
% OUTPUT-NEXT:already ended: noproc
% OUTPUT-NEXT:{{(first|second)}}: domain_error("int.\'//\': division by zero")
% OUTPUT-NEXT:{{(first|second)}}: domain_error("int.\'//\': division by zero")
% OUTPUT-NEXT:different refs
% OUTPUT-NEXT:kept: normal
% OUTPUT-NEXT:2
% OUTPUT-NEXT:constant: normal
% OUTPUT-NEXT:request: noproc
% OUTPUT-NOT:{{.}}
