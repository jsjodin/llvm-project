% DESIGN: Distribution and wire format / Remote calls
% Remote calls are requests and replies over send and receive, as between
% local processes, and work across nodes because mailbox handles do. A
% blocking request gets its reply, or timeout when the server is slow; a
% request sent with send_request is collected later; and a request to a
% server whose node quits before it replies is down(noconnection). The
% oracle is a translation into distributed Erlang, with request written as
% the design writes it: a monitor, a send and a receive.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/remote_calls_translation.erl
% NODES: server client
%
% RUN: merkc -O0 -fprocess-declarations %s -o %t0
% RUN: %nodes %t0 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %nodes %t0 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2
% RUN: %nodes %t2 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:server: quitting without a reply
% OUTPUT-NEXT:client: add: reply(5)
% OUTPUT-NEXT:client: slow, waiting 100 ms: timeout
% OUTPUT-NEXT:client: sent a request
% OUTPUT-NEXT:client: collected: message(42)
% OUTPUT-NEXT:client: quit: down(noconnection)
% OUTPUT-NOT:{{.}}
:- module run_remote_calls.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process, string.

:- type call
    --->    add(int, int, mailbox(int))
    ;       slow(int, mailbox(int))
    ;       quit(mailbox(int)).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = ["server"] then
        server(!IO)
    else if Args = ["client", Address] then
        client(Address, !IO)
    else
        io.write_string("usage: server | client ADDRESS\n", !IO)
    ).

:- pred server(io::di, io::uo) is det.

server(!IO) :-
    listen("127.0.0.1:0", Listened, !IO),
    (
        Listened = ok(Port),
        io.format("listening 127.0.0.1:%d\n", [i(Port)], !IO),
        io.flush_output(!IO),
        new_mailbox(Calls, !IO),
        register("calls", Calls, _, !IO),
        serve(Calls, !IO)
    ;
        Listened = error(Message),
        io.write_string(Message ++ "\n", !IO)
    ).

:- pred serve(mailbox(call)::in, io::di, io::uo) is det.

serve(Calls, !IO) :-
    receive(Calls, any, infinity, R, !IO),
    (
        R = message(add(X, Y, Reply)),
        send(Reply, X + Y, !IO),
        serve(Calls, !IO)
    ;
        R = message(slow(Milliseconds, Reply)),
        % Waits on a mailbox that nothing sends to.
        new_mailbox(Nothing, !IO),
        receive(Nothing : mailbox(int), any, after(Milliseconds), _, !IO),
        send(Reply, Milliseconds, !IO),
        serve(Calls, !IO)
    ;
        R = message(quit(_)),
        io.write_string("quitting without a reply\n", !IO)
    ;
        R = timeout
    ).

:- pred client(string::in, io::di, io::uo) is det.

client(Address, !IO) :-
    connect(Address, Connected, !IO),
    (
        Connected = ok(Node),
        lookup(Node, "calls", after(5000), Found, !IO),
        (
            Found = ok(Calls),
            calls(Calls, !IO)
        ;
            Found = error(Message),
            io.write_string(Message ++ "\n", !IO)
        )
    ;
        Connected = error(Message),
        io.write_string(Message ++ "\n", !IO)
    ).

:- pred calls(mailbox(call)::in, io::di, io::uo) is det.

calls(Calls, !IO) :-
    request(Calls, (func(Reply) = add(2, 3, Reply)), after(5000), R1, !IO),
    io.write_string("add: ", !IO),
    io.write_line(R1, !IO),
    request(Calls, (func(Reply) = slow(1000, Reply)), after(100), R2, !IO),
    io.write_string("slow, waiting 100 ms: ", !IO),
    io.write_line(R2, !IO),
    send_request(Calls, (func(Reply) = add(40, 2, Reply)), Pending, !IO),
    io.write_string("sent a request\n", !IO),
    % The slow call's late reply went to its own mailbox, not this one.
    receive(Pending, any, after(5000), R3, !IO),
    io.write_string("collected: ", !IO),
    io.write_line(R3, !IO),
    request(Calls, (func(Reply) = quit(Reply)), after(5000), R4, !IO),
    io.write_string("quit: ", !IO),
    io.write_line(R4, !IO).
