% The mailboxes of one process (library/process.m): messages arrive in
% order; a selector takes the oldest message it accepts and leaves the
% others queued; polling and a timeout give no when nothing is queued; a
% message can hold a mailbox, which a reply is sent to; a closed mailbox
% drops what is queued and what is sent to it, and its handle does not
% reach a later mailbox in the same slot; and many messages, and a long
% list, which the copy loops over rather than recursing. mmc accepts the
% program with process.m, but has no process module to run it with, so the
% output is written by hand.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_local.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, maybe, process, string.

:- type shape
    --->    circle(float)
    ;       rectangle(int, int)
    ;       named(string, shape)
    ;       origin.

:- type request
    --->    add(int, int)
    ;       reply_to(mailbox(int))
    ;       stop.

:- pred show(maybe(T)::in, io::di, io::uo) is det.
show(M, !IO) :-
    io.write(M, !IO),
    io.nl(!IO).

:- pred big(int::in, int::out) is semidet.
big(N, N) :-
    N > 10.

main(!IO) :-
    % Messages arrive in order.
    new_mailbox(Ints, !IO),
    send(Ints, 1, !IO),
    send(Ints, 2, !IO),
    send(Ints, 3, !IO),
    receive(Ints, any, infinity, A, !IO),
    show(A, !IO),
    receive(Ints, any, after(0), B, !IO),
    show(B, !IO),
    % Selective: the first message the selector accepts, the rest stay.
    send(Ints, 20, !IO),
    send(Ints, 4, !IO),
    receive(Ints, big, after(0), C, !IO),
    show(C, !IO),
    receive(Ints, any, after(0), D, !IO),
    show(D, !IO),
    receive(Ints, any, after(0), E, !IO),
    show(E, !IO),
    % Nothing left: polling and a timeout give no.
    receive(Ints, any, after(0), F, !IO),
    show(F, !IO),
    receive(Ints, any, after(20), G, !IO),
    show(G, !IO),
    % Terms, strings, lists, arrays and floats are copied whole.
    new_mailbox(Shapes, !IO),
    send(Shapes, named("box", rectangle(3, 4)), !IO),
    send(Shapes, circle(1.5), !IO),
    send(Shapes, origin, !IO),
    receive(Shapes, any, after(0), S1, !IO),
    show(S1, !IO),
    receive(Shapes, any, after(0), S2, !IO),
    show(S2, !IO),
    receive(Shapes, any, after(0), S3, !IO),
    show(S3, !IO),
    new_mailbox(Lists, !IO),
    send(Lists, ["a", "b", "c"], !IO),
    receive(Lists, any, after(0), L, !IO),
    show(L, !IO),
    new_mailbox(Arrays, !IO),
    Array0 = array([1.0, 2.5, 4.0]),
    send(Arrays, Array0, !IO),
    receive(Arrays, any, after(0), Array, !IO),
    show(Array, !IO),
    % A message that holds a mailbox: the reply goes to it.
    new_mailbox(Requests, !IO),
    new_mailbox(Replies, !IO),
    send(Requests, add(2, 3), !IO),
    send(Requests, reply_to(Replies), !IO),
    send(Requests, stop, !IO),
    serve(Requests, 0, !IO),
    receive(Replies, any, after(0), Reply, !IO),
    show(Reply, !IO),
    % A closed mailbox drops what is queued and what is sent to it.
    send(Ints, 5, !IO),
    close_mailbox(Ints, !IO),
    send(Ints, 6, !IO),
    receive(Ints, any, after(0), H, !IO),
    show(H, !IO),
    % A new mailbox in the same slot is not reached through the old handle.
    new_mailbox(Fresh, !IO),
    send(Ints, 7, !IO),
    send(Fresh, 8, !IO),
    receive(Fresh, any, after(0), I, !IO),
    show(I, !IO),
    receive(Fresh, any, after(0), J, !IO),
    show(J, !IO),
    % Many messages, and a long list, which the copy loops over.
    new_mailbox(Many, !IO),
    send_all(Many, 1, 100000, !IO),
    sum_all(Many, 0, Sum, !IO),
    io.write_int(Sum, !IO),
    io.nl(!IO),
    new_mailbox(Long, !IO),
    send(Long, upto(1, 1000000, []), !IO),
    receive(Long, any, after(0), MaybeLong, !IO),
    (
        MaybeLong = yes(Received),
        io.write_int(list.length(Received), !IO)
    ;
        MaybeLong = no,
        io.write_string("none", !IO)
    ),
    io.nl(!IO).

:- pred serve(mailbox(request)::in, int::in, io::di, io::uo) is det.
serve(Requests, Total, !IO) :-
    receive(Requests, any, after(0), MaybeRequest, !IO),
    (
        MaybeRequest = yes(add(X, Y)),
        serve(Requests, Total + X + Y, !IO)
    ;
        MaybeRequest = yes(reply_to(Box)),
        send(Box, Total, !IO),
        serve(Requests, Total, !IO)
    ;
        MaybeRequest = yes(stop)
    ;
        MaybeRequest = no
    ).

:- pred send_all(mailbox(int)::in, int::in, int::in, io::di, io::uo) is det.
send_all(Box, I, N, !IO) :-
    ( if I =< N then
        send(Box, I, !IO),
        send_all(Box, I + 1, N, !IO)
    else
        true
    ).

:- pred sum_all(mailbox(int)::in, int::in, int::out, io::di, io::uo) is det.
sum_all(Box, Sum0, Sum, !IO) :-
    receive(Box, any, after(0), M, !IO),
    (
        M = yes(N),
        sum_all(Box, Sum0 + N, Sum, !IO)
    ;
        M = no,
        Sum = Sum0
    ).

    % The numbers from Low to High, in order.
:- func upto(int, int, list(int)) = list(int).
upto(Low, High, Acc) =
    ( if High < Low then Acc else upto(Low, High - 1, [High | Acc]) ).

% OUTPUT:yes(1)
% OUTPUT-NEXT:yes(2)
% OUTPUT-NEXT:yes(20)
% OUTPUT-NEXT:yes(3)
% OUTPUT-NEXT:yes(4)
% OUTPUT-NEXT:no
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes(named("box", rectangle(3, 4)))
% OUTPUT-NEXT:yes(circle(1.5))
% OUTPUT-NEXT:yes(origin)
% OUTPUT-NEXT:yes(["a", "b", "c"])
% OUTPUT-NEXT:yes(array([1.0, 2.5, 4.0]))
% OUTPUT-NEXT:yes(5)
% OUTPUT-NEXT:no
% OUTPUT-NEXT:yes(8)
% OUTPUT-NEXT:no
% OUTPUT-NEXT:5000050000
% OUTPUT-NEXT:1000000
% OUTPUT-EMPTY:
