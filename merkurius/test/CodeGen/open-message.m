% Messages with open terms in them (docs/INPROGRESS.md, Open types, step
% 6), sent by open.send and received by open.receive. A message is copied
% as copy_term copies a term: bound variables as their values, and each
% unbound variable as one new variable, however often the message holds
% it, so that binding it in one place of the copy binds every place. The
% sender's variables stay its own: neither side's later bindings reach the
% other. A cyclic message is copied with its cycle, and a ground open term
% sent by process.send is received too. mmc has no open types, so the
% outputs are checked by hand: each line says what it shows. A hole is
% bound by unifying the term around it, as in R = message([5 | _]): X = 5
% would make X an int, taken out of its field, which fails while unbound.
%
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module open_message.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, open, process.

    % Eight constructors with arguments: the open layout shares tag 6 for
    % the last three, where the ordinary one shares 7 for the last two.
:- type eight ---> p0(int) ; p1(int) ; p2(int) ; p3(int) ; p4(int)
    ; p5(int) ; p6(int) ; p7(int).
:- type item ---> item(string, float, closed(list(int)), eight).

main(!IO) :-
    twice(!IO), bound(!IO), layouts(!IO), cyclic(!IO), ground_send(!IO),
    process(!IO).

    % The one message in the mailbox, or nothing.
:- pred take(mailbox(T)::in, received(T)::oa, io::di, io::uo) is det.
take(Box, R, !IO) :-
    open.receive(Box, after(0), R, !IO).

    % The same variable twice: the copy has one variable in both places,
    % which a binding of the first fills in the second too; the sender's
    % list keeps its own unbound variables, and binding them later does
    % not reach the copy.
:- pred twice(io::di, io::uo) is det.
twice(!IO) :-
    promise_pure (
        new_mailbox(Box, !IO),
        L = [A, A, _] : open(list(int)),
        impure open.send(Box, L, !IO),
        take(Box, R, !IO),
        impure open.write(R, !IO), io.nl(!IO),
        ( if R = message([5 | _]) then
            impure open.write(R, !IO), io.nl(!IO)
        else
            io.write_string("twice: no\n", !IO)
        ),
        impure open.write(L, !IO), io.nl(!IO),
        ( if L = [9 | _] then
            impure open.write({L, R}, !IO), io.nl(!IO)
        else
            io.write_string("twice: no\n", !IO)
        )
    ).
% OUTPUT: message([_1, _1, _2])
% OUTPUT-NEXT: message([5, 5, _1])
% OUTPUT-NEXT: [_1, _1, _2]
% OUTPUT-NEXT: {[9, 9, _1], message([5, 5, _2])}

    % Bound variables, through a chain, are sent as their values.
:- pred bound(io::di, io::uo) is det.
bound(!IO) :-
    promise_pure (
        new_mailbox(Box, !IO),
        L = [A, B | T] : open(list(int)),
        A = B, B = 3, T = [_],
        impure open.send(Box, L, !IO),
        take(Box, R, !IO),
        impure open.write(R, !IO), io.nl(!IO)
    ).
% OUTPUT-NEXT: message([3, 3, _1])

    % Every field's encoding, bound and unbound, with the variables shared
    % across two messages kept apart: each message is its own copy.
:- pred layouts(io::di, io::uo) is det.
layouts(!IO) :-
    promise_pure (
        new_mailbox(Box, !IO),
        I = [item("a", 1.5, closed([1, 2]), p6(6)), item(S, F, C, p7(E)),
            item(S, F, C, p5(E))] : open(list(item)),
        impure open.send(Box, I, !IO),
        impure open.send(Box, I, !IO),
        take(Box, R1, !IO),
        take(Box, R2, !IO),
        ( if
            R1 = message([_, item("s", _, _, _) | _]),
            R2 = message([_, _, item(_, 2.5, _, _)])
        then
            impure open.write(R1, !IO), io.nl(!IO),
            impure open.write(R2, !IO), io.nl(!IO)
        else
            io.write_string("layouts: no\n", !IO)
        )
    ).
% OUTPUT-NEXT: message([item("a", 1.5, closed([1, 2]), p6(6)), item("s", _1, _2, p7(_3)), item("s", _1, _2, p5(_3))])
% OUTPUT-NEXT: message([item("a", 1.5, closed([1, 2]), p6(6)), item(_1, 2.5, _2, p7(_3)), item(_1, 2.5, _2, p5(_3))])

    % A cyclic message is copied with its cycle, and the copy ends.
:- pred cyclic(io::di, io::uo) is det.
cyclic(!IO) :-
    promise_pure (
        new_mailbox(Box, !IO),
        ( if L = [1, _ | T] : open(list(int)), T = L then
            impure open.send(Box, L, !IO),
            take(Box, R, !IO),
            impure open.write(R, !IO), io.nl(!IO),
            ( if R = message([_, 2 | _]) then
                impure open.write(R, !IO), io.nl(!IO)
            else
                io.write_string("cyclic: no\n", !IO)
            )
        else
            io.write_string("cyclic: no\n", !IO)
        )
    ).
% OUTPUT-NEXT: message([1, _1 | ...])
% OUTPUT-NEXT: message([1, 2 | ...])

    % A ground open term sent by process.send.
:- pred ground_send(io::di, io::uo) is det.
ground_send(!IO) :-
    promise_pure (
        new_mailbox(Box, !IO),
        process.send(Box, to_open([7, 8]), !IO),
        take(Box, R, !IO),
        impure open.write(R, !IO), io.nl(!IO)
    ).
% OUTPUT-NEXT: message([7, 8])

    % Between processes: a child receives a list with a variable twice,
    % binds it and sends the list back, which fills both places in the
    % reply; the parent's list keeps its own variables.
:- pred doubler(mailbox(open(list(int)))::in, mailbox(open(list(int)))::in,
    io::di, io::uo) is det.
doubler(Reply, Box, !IO) :-
    promise_pure (
        open.receive(Box, infinity, R, !IO),
        ( if R = message(L), L = [4 | _] then
            impure open.send(Reply, L, !IO)
        else
            impure open.send(Reply, [] : open(list(int)), !IO)
        )
    ).

:- pred process(io::di, io::uo) is det.
process(!IO) :-
    promise_pure (
        new_mailbox(Reply, !IO),
        spawn(doubler(Reply), Child, !IO),
        L = [A, _, A] : open(list(int)),
        impure open.send(Child, L, !IO),
        open.receive(Reply, infinity, R, !IO),
        impure open.write(R, !IO), io.nl(!IO),
        impure open.write(L, !IO), io.nl(!IO)
    ).
% OUTPUT-NEXT: message([4, _1, 4])
% OUTPUT-NEXT: [_1, _2, _1]
