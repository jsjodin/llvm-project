% DESIGN: Processes and mailboxes
% A mailbox is a FIFO: receive with any takes the oldest message. A
% selective receive takes the oldest message its selector accepts, and
% leaves the others queued in their order. after(0) polls, and is timeout
% on an empty mailbox. A selector over several mailboxes takes the
% matching message that arrived first, across them, as Erlang's receive
% takes the first message in the queue that matches any clause; its
% cases' order only decides which case maps a message.
% TRANSLATION: Inputs/mailbox_order_translation.erl
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[data(1), data(2), data(3)]
% OUTPUT-NEXT:empty: timeout
% OUTPUT-NEXT:urgent: message(10)
% OUTPUT-NEXT:urgent: message(11)
% OUTPUT-NEXT:urgent: timeout
% OUTPUT-NEXT:[data(4), data(5), done]
% OUTPUT-NEXT:[from_a(data(6)), from_b(2), from_a(data(7)), from_b(4)]
% OUTPUT-NEXT:left in b: message(1)
% OUTPUT-NOT:{{.}}
:- module run_mailbox_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- type msg
    --->    data(int)
    ;       urgent(int)
    ;       done.

:- type arrival
    --->    from_a(msg)
    ;       from_b(int).

:- pred is_urgent(msg::in, int::out) is semidet.

is_urgent(urgent(N), N).

:- pred drain(mailbox(msg)::in, list(msg)::in, list(msg)::out,
    io::di, io::uo) is det.

drain(Box, !Got, !IO) :-
    receive(Box, any, after(0), R, !IO),
    (
        R = message(M),
        drain(Box, [M | !.Got], !:Got, !IO)
    ;
        R = timeout,
        !:Got = list.reverse(!.Got)
    ).

:- pred show(string::in, received(T)::in, io::di, io::uo) is det.

show(Label, R, !IO) :-
    io.write_string(Label ++ ": ", !IO),
    io.write_line(R, !IO).

main(!IO) :-
    new_mailbox(Box, !IO),
    % FIFO.
    list.foldl(send(Box), [data(1), data(2), data(3)], !IO),
    drain(Box, [], Got1, !IO),
    io.write_line(Got1, !IO),
    % Polling an empty mailbox.
    receive(Box, any, after(0), R0, !IO),
    show("empty", R0, !IO),
    % A selective receive takes the oldest urgent message and leaves the
    % rest in order; a second takes the next urgent one.
    list.foldl(send(Box),
        [data(4), urgent(10), data(5), urgent(11), done], !IO),
    receive(Box, is_urgent, after(0), R1, !IO),
    show("urgent", R1, !IO),
    receive(Box, is_urgent, after(0), R2, !IO),
    show("urgent", R2, !IO),
    receive(Box, is_urgent, after(0), R3, !IO),
    show("urgent", R3, !IO),
    drain(Box, [], Got2, !IO),
    io.write_line(Got2, !IO),
    % A selector over two mailboxes: the oldest message across them that
    % a case accepts. B's case accepts only even numbers.
    new_mailbox(BoxB, !IO),
    send(BoxB, 1, !IO),
    send(Box, data(6), !IO),
    send(BoxB, 2, !IO),
    send(Box, data(7), !IO),
    send(BoxB, 4, !IO),
    Sel = selecting_matching(BoxB, even_b,
        selecting(Box, (func(M) = from_a(M)), new_selector)),
    receive_all(Sel, [], Events, !IO),
    io.write_line(Events, !IO),
    % B's odd message is still queued.
    receive(BoxB, any, after(0), R4, !IO),
    show("left in b", R4, !IO).

:- pred even_b(int::in, arrival::out) is semidet.

even_b(N, from_b(N)) :-
    N mod 2 = 0.

:- pred receive_all(selector(arrival)::in,
    list(arrival)::in, list(arrival)::out, io::di, io::uo) is det.

receive_all(Sel, !Events, !IO) :-
    receive(Sel, after(0), R, !IO),
    (
        R = message(E),
        receive_all(Sel, [E | !.Events], !:Events, !IO)
    ;
        R = timeout,
        !:Events = list.reverse(!.Events)
    ).
