% DESIGN: Processes and mailboxes / Failure handling
% An uncaught exception ends the process that raised it, with the report
% of the failure on standard error, and its monitors deliver
% failed(Text), the exception as written; the other processes go on. A
% request to a server that fails before it replies is down(failed(Text)).
% TRANSLATION: Inputs/process_failure_translation.erl
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:Uncaught Merkurius exception:
% OUTPUT-NEXT:"too big: 5"
% OUTPUT-NEXT:failed: "too big: 5"
% OUTPUT-NEXT:main goes on
% OUTPUT-NEXT:reply(8)
% OUTPUT-NEXT:Uncaught Merkurius exception:
% OUTPUT-NEXT:40
% OUTPUT-NEXT:down, failed: 40
% OUTPUT-NOT:{{.}}
:- module run_process_failure.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exception, int, list, process, string.

:- type request
    --->    double(int, mailbox(int)).

:- pred thrower(mailbox(int)::in, io::di, io::uo) is det.

thrower(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N > 0 then
        throw("too big: " ++ string.int_to_string(N))
    else
        true
    ).

    % Doubles numbers up to 10, and fails on a larger one.
:- pred doubler(mailbox(request)::in, io::di, io::uo) is det.

doubler(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(double(N, ReplyTo)),
        ( if N > 10 then
            throw(N)
        else
            send(ReplyTo, 2 * N, !IO),
            doubler(Self, !IO)
        )
    ;
        R = timeout
    ).

:- func id(down_info) = down_info.

id(D) = D.

:- func ask(int, mailbox(int)) = request.

ask(N, ReplyTo) = double(N, ReplyTo).

:- pred show_reason(down_reason::in, io::di, io::uo) is det.

show_reason(Reason, !IO) :-
    ( if Reason = failed(Text) then
        io.format("failed: %s\n", [s(Text)], !IO)
    else
        io.write_line(Reason, !IO)
    ).

main(!IO) :-
    new_mailbox(Downs, !IO),
    spawn(thrower, T, !IO),
    monitor(T, Downs, id, _, !IO),
    send(T, 5, !IO),
    receive(Downs, any, infinity, R, !IO),
    (
        R = message(D),
        show_reason(reason(D), !IO)
    ;
        R = timeout,
        io.write_string("timeout\n", !IO)
    ),
    io.write_string("main goes on\n", !IO),
    spawn(doubler, S, !IO),
    request(S, ask(4), infinity, Reply1, !IO),
    io.write_line(Reply1, !IO),
    request(S, ask(40), infinity, Reply2, !IO),
    (
        Reply2 = down(Reason),
        io.write_string("down, ", !IO),
        show_reason(Reason, !IO)
    ;
        ( Reply2 = reply(_) ; Reply2 = timeout ),
        io.write_line(Reply2, !IO)
    ).
