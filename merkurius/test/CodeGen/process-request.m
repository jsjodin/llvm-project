% request (design document, "Request and reply"): a reply; a timeout, after
% which the late reply goes nowhere; a server that fails while it is asked and
% one that has ended; a reply that comes before its server ends; and
% send_request, whose reply is received from a selector.
%
% RUN: merkc -O0 %s -o %t.0 && %t.0 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.2 && %t.2 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_WORKERS=1 %t.2 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_request.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process, string.

:- type ask
    --->    double(int, mailbox(int))
    ;       slow(int, mailbox(int))
    ;       divide(int, mailbox(int))
    ;       last(int, mailbox(int))
    ;       stop.

:- pred server(mailbox(ask)::in, io::di, io::uo) is det.
server(Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = message(double(N, To)),
        send(To, 2 * N, !IO),
        server(Box, !IO)
    ;
        Got = message(slow(N, To)),
        new_mailbox(Idle : mailbox(int), !IO),
        receive(Idle, any, after(100), _, !IO),
        close_mailbox(Idle, !IO),
        send(To, N, !IO),
        server(Box, !IO)
    ;
        Got = message(divide(N, To)),
        send(To, 100 // N, !IO),
        server(Box, !IO)
    ;
        Got = message(last(N, To)),
        send(To, N, !IO)
    ;
        Got = message(stop)
    ;
        Got = timeout
    ).

:- pred show(reply(int)::in, io::di, io::uo) is det.
show(reply(N), !IO) :-
    io.write_string("reply " ++ int_to_string(N) ++ "\n", !IO).
show(timeout, !IO) :-
    io.write_string("timeout\n", !IO).
show(down(Reason), !IO) :-
    io.write_string("down: " ++ reason_text(Reason) ++ "\n", !IO).

:- pred many(int::in, mailbox(ask)::in, int::in, int::out, io::di, io::uo)
    is det.
many(N, Server, !Sum, !IO) :-
    ( if N = 0 then
        true
    else
        request(Server, (func(R) = double(N, R)), infinity, Got, !IO),
        ( if Got = reply(V) then
            !:Sum = !.Sum + V
        else
            true
        ),
        many(N - 1, Server, !Sum, !IO)
    ).

main(!IO) :-
    spawn(server, Server, !IO),
    request(Server, (func(R) = double(21, R)), after(10000), First, !IO),
    show(First, !IO),

    % Too slow, and what it sends later goes nowhere.
    request(Server, (func(R) = slow(1, R)), after(1), Slow, !IO),
    show(Slow, !IO),
    request(Server, (func(R) = double(4, R)), after(10000), Next, !IO),
    show(Next, !IO),

    % Many, each with its own mailboxes and monitor.
    many(2000, Server, 0, Sum, !IO),
    io.write_int(Sum, !IO),
    io.nl(!IO),

    % The reply comes first, though the server ends straight after it.
    spawn(server, Ending, !IO),
    request(Ending, (func(R) = last(5, R)), infinity, Last, !IO),
    show(Last, !IO),

    % It has ended, or is ending.
    spawn(server, Stopped, !IO),
    new_mailbox(Downs : mailbox(down_info), !IO),
    monitor(Stopped, Downs, (func(D) = D), _, !IO),
    send(Stopped, stop, !IO),
    receive(Downs, any, infinity, _, !IO),
    request(Stopped, (func(R) = double(1, R)), after(10000), Dead, !IO),
    show(Dead, !IO),

    % The server fails with the request in hand.
    spawn(server, Failing, !IO),
    request(Failing, (func(R) = divide(0, R)), infinity, Failed, !IO),
    show(Failed, !IO),

    % send_request, with the reply taken from a selector.
    send_request(Server, (func(R) = double(50, R)), ReplyTo, !IO),
    Selector = selecting(ReplyTo, (func(N) = N), new_selector),
    receive(Selector, after(10000), Selected, !IO),
    close_mailbox(ReplyTo, !IO),
    io.write_line(Selected, !IO).

% OUTPUT:reply 42
% OUTPUT-NEXT:timeout
% OUTPUT-NEXT:reply 8
% OUTPUT-NEXT:4002000
% OUTPUT-NEXT:reply 5
% OUTPUT-NEXT:down: noproc
% OUTPUT-NEXT:down: domain_error("int.\'//\': division by zero")
% OUTPUT-NEXT:message(100)
% OUTPUT-NOT:{{.}}
