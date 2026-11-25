% DESIGN: Processes and mailboxes / The receive goal
% A receive goal takes the oldest message, across its mailboxes, that
% some arm selects, whatever the order of the arms; a mailbox's own arms
% are tried in the order written, so a message two select goes to the
% first. A message no arm selects stays queued for a later receive. A
% pattern's variable that occurs outside the goal is tested, the others
% are the arm's own. The timeout arm runs when no message is selected in
% time.
% MERKC-FLAGS: -freceive-goal
% TRANSLATION: Inputs/receive_goal_translation.erl
%
% RUN: merkc -O0 -freceive-goal %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -freceive-goal %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:["b1", "a:two", "b3", "a:four"]
% OUTPUT-NEXT:small(3)
% OUTPUT-NEXT:large(30)
% OUTPUT-NEXT:large(3)
% OUTPUT-NEXT:selected 8
% OUTPUT-NEXT:left queued: 7
% OUTPUT-NEXT:reply to 2: 15
% OUTPUT-NEXT:reply to 1: 5
% OUTPUT-NEXT:["hello", "world"]
% OUTPUT-NEXT:timeout
% OUTPUT-NOT:{{.}}
:- module run_receive_goal.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- type counter_msg
    --->    increment(int)
    ;       get(int, mailbox(reply))
    ;       notes(mailbox(list(string)))
    ;       stop.

:- type reply
    --->    reply(int, int).                % the request's number, the count

:- type note
    --->    note(string).

    % A server with two mailboxes: requests, and notes it collects.
:- pred server(int::in, list(string)::in, mailbox(counter_msg)::in,
    mailbox(note)::in, io::di, io::uo) is det.

server(Count, Notes, Self, NoteBox, !IO) :-
    receive [io(!IO)] (
        select(NoteBox, note(Text)) ->
            server(Count, [Text | Notes], Self, NoteBox, !IO)
    ;   select(Self, increment(N)) ->
            server(Count + N, Notes, Self, NoteBox, !IO)
    ;   select(Self, get(Id, ReplyTo)) ->
            send(ReplyTo, reply(Id, Count), !IO),
            server(Count, Notes, Self, NoteBox, !IO)
    ;   select(Self, notes(ReplyTo)) ->
            send(ReplyTo, list.reverse(Notes), !IO),
            server(Count, Notes, Self, NoteBox, !IO)
    ;   select(Self, stop) ->
            true
    ).

:- type small_or_large
    --->    small(int)
    ;       large(int).

main(!IO) :-
    % Two mailboxes of main's own, filled before any receive.
    new_mailbox(A, !IO),
    new_mailbox(B, !IO),
    send(B, 1, !IO),
    send(A, "two", !IO),
    send(B, 3, !IO),
    send(A, "four", !IO),
    take_oldest(A, B, 4, [], Oldest, !IO),
    io.write_line(Oldest, !IO),
    % A mailbox's arms in order: the first arm that selects a message
    % takes it.
    new_mailbox(C, !IO),
    list.foldl(send(C), [3, 30, 3], !IO),
    three_first(C, !IO),
    three_first(C, !IO),
    any_first(C, !IO),
    % Selective: 7 is not selected and stays queued.
    send(C, 7, !IO),
    send(C, 8, !IO),
    receive [io(!IO)] (
        select(C, 8) -> io.write_string("selected 8\n", !IO)
    ),
    receive [timeout(after(0)), io(!IO)] (
        select(C, M) -> io.format("left queued: %d\n", [i(M)], !IO)
    ;   timeout -> io.write_string("nothing left\n", !IO)
    ),
    % A spawned server, which makes its notes mailbox and sends it back,
    % as only a mailbox's owner may receive from it; replies taken out of
    % order by a tested variable.
    new_mailbox(Setup, !IO),
    spawn(server_with(Setup), S, !IO),
    % Notes occurs after the goal, so the pattern would test it: the arm's
    % own variable binds it in the arm's goal.
    receive [io(!IO)] (
        select(Setup, Box) -> Notes = Box
    ),
    send(S, increment(5), !IO),
    send(Notes, note("hello"), !IO),
    new_mailbox(Replies, !IO),
    send(S, get(1, Replies), !IO),
    send(S, increment(10), !IO),
    send(S, get(2, Replies), !IO),
    reply_to(Replies, 2, !IO),
    reply_to(Replies, 1, !IO),
    send(Notes, note("world"), !IO),
    new_mailbox(NotesBack, !IO),
    send(S, notes(NotesBack), !IO),
    receive [io(!IO)] (
        select(NotesBack, Texts) -> io.write_line(Texts, !IO)
    ),
    send(S, stop, !IO),
    % A timeout arm with nothing to take.
    receive [timeout(after(10)), io(!IO)] (
        select(Replies, _) -> io.write_string("a stray reply\n", !IO)
    ;   timeout -> io.write_string("timeout\n", !IO)
    ).

:- pred server_with(mailbox(mailbox(note))::in, mailbox(counter_msg)::in,
    io::di, io::uo) is det.

server_with(Setup, Self, !IO) :-
    new_mailbox(Notes, !IO),
    send(Setup, Notes, !IO),
    server(0, [], Self, Notes, !IO).

:- pred take_oldest(mailbox(string)::in, mailbox(int)::in, int::in,
    list(string)::in, list(string)::out, io::di, io::uo) is det.

take_oldest(A, B, K, !Got, !IO) :-
    ( if K =< 0 then
        !:Got = list.reverse(!.Got)
    else
        % B's arm is written first, A's second: the order across mailboxes
        % is the messages'.
        receive [io(!IO)] (
            select(B, N) -> Text = "b" ++ string.int_to_string(N)
        ;   select(A, S) -> Text = "a:" ++ S
        ),
        take_oldest(A, B, K - 1, [Text | !.Got], !:Got, !IO)
    ).

:- pred three_first(mailbox(int)::in, io::di, io::uo) is det.

three_first(C, !IO) :-
    receive [io(!IO)] (
        select(C, 3) -> io.write_line(small(3), !IO)
    ;   select(C, N) -> io.write_line(large(N), !IO)
    ).

    % The same arms the other way round: the first takes the 3.
:- pred any_first(mailbox(int)::in, io::di, io::uo) is det.

any_first(C, !IO) :-
    receive [io(!IO)] (
        select(C, N) -> io.write_line(large(N), !IO)
    ;   select(C, 3) -> io.write_line(small(3), !IO)
    ).

:- pred reply_to(mailbox(reply)::in, int::in, io::di, io::uo) is det.

reply_to(Replies, Id, !IO) :-
    receive [io(!IO)] (
        select(Replies, reply(Id, Count)) ->
            io.format("reply to %d: %d\n", [i(Id), i(Count)], !IO)
    ).
