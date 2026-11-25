% DESIGN: Processes and mailboxes
% A spawned process runs its predicate on a new mailbox. request sends a
% request made with a fresh reply mailbox and waits for the reply, the
% timeout, or the server's end: reply(N), timeout or down(Reason).
% send_request returns the reply mailbox instead, for the caller to
% receive from and close. A monitor delivers down_info through the
% caller's function when the watched process ends: normal when it
% returns, killed when stop ends it, and noproc at once when it had
% ended already. A request to a server that has ended is down(noproc).
% TRANSLATION: Inputs/spawn_request_translation.erl
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:reply(6)
% OUTPUT-NEXT:message(10)
% OUTPUT-NEXT:returned: normal
% OUTPUT-NEXT:ended already: noproc
% OUTPUT-NEXT:down(noproc)
% OUTPUT-NEXT:timeout
% OUTPUT-NEXT:stopped: killed
% OUTPUT-NOT:{{.}}
:- module run_spawn_request.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process, string.

:- type counter_msg
    --->    increment(int)
    ;       get(mailbox(int))
    ;       stop.

:- pred counter(int::in, mailbox(counter_msg)::in, io::di, io::uo) is det.

counter(N, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(increment(K)),
        counter(N + K, Self, !IO)
    ;
        R = message(get(ReplyTo)),
        send(ReplyTo, N, !IO),
        counter(N, Self, !IO)
    ;
        R = message(stop)
    ;
        R = timeout
    ).

    % Takes requests and never replies.
:- pred silent(mailbox(counter_msg)::in, io::di, io::uo) is det.

silent(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(_),
        silent(Self, !IO)
    ;
        R = timeout
    ).

:- func id(down_info) = down_info.

id(D) = D.

:- func make_get(mailbox(int)) = counter_msg.

make_get(R) = get(R).

:- pred show_down(string::in, mailbox(down_info)::in, io::di, io::uo) is det.

show_down(Label, Downs, !IO) :-
    receive(Downs, any, infinity, R, !IO),
    io.write_string(Label ++ ": ", !IO),
    (
        R = message(D),
        io.write_line(reason(D), !IO)
    ;
        R = timeout,
        io.write_string("timeout\n", !IO)
    ).

main(!IO) :-
    spawn(counter(0), C, !IO),
    send(C, increment(1), !IO),
    send(C, increment(2), !IO),
    send(C, increment(3), !IO),
    request(C, make_get, infinity, Reply1, !IO),
    io.write_line(Reply1, !IO),
    send(C, increment(4), !IO),
    send_request(C, make_get, ReplyBox, !IO),
    receive(ReplyBox, any, infinity, Got, !IO),
    close_mailbox(ReplyBox, !IO),
    io.write_line(Got, !IO),
    % The counter returns on stop: normal.
    new_mailbox(Downs, !IO),
    monitor(C, Downs, id, _, !IO),
    send(C, stop, !IO),
    show_down("returned", Downs, !IO),
    % Watching it again, after it has ended: noproc.
    monitor(C, Downs, id, _, !IO),
    show_down("ended already", Downs, !IO),
    request(C, make_get, infinity, Reply2, !IO),
    io.write_line(Reply2, !IO),
    % A server that never replies: the request times out.
    spawn(silent, S, !IO),
    request(S, make_get, after(50), Reply3, !IO),
    io.write_line(Reply3 : reply(int), !IO),
    % stop ends it: killed.
    monitor(S, Downs, id, _, !IO),
    stop(S, !IO),
    show_down("stopped", Downs, !IO).
