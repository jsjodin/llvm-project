% Receive goals on the host, as lowering's loop over the runtime's
% candidates (Lowering.cpp, LowerSelect; runtime/mailbox.cpp,
% merk_select_next):
%
% - What a monitor left becomes the message, by the monitor's function,
%   when a receive goal reaches it, past messages that no arm selects.
% - A matcher's captured value on the heap, a key made at run time, and
%   what is live across the receive, a list, survive collections at every
%   allocation, which the matchers' results make (the stack maps of the
%   loop's calls).
% - Two variables of one mailbox are two cases, which try each message in
%   turn, so the oldest message goes first, to whichever case selects it.
% - A message skipped by many receives is still taken in order.
%
% mmc has no receive goal, so the output is written by hand.
%
% RUN: merkc -freceive-goal -fprocess-declarations -Wno-unselected-message -O0 %s -o %t0
% RUN: %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -freceive-goal -fprocess-declarations -Wno-unselected-message -O2 %s -o %t2
% RUN: %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
%
% CHECK:down: normal
% CHECK-NEXT:then: note(1)
% CHECK-NEXT:then: note(2)
% CHECK-NEXT:key a: [2, 4, 6]
% CHECK-NEXT:key b: [1, 3, 5]
% CHECK-NEXT:aliases: [large(5), small(6), large(7)]
% CHECK-NEXT:last: 1000

:- module receive_goal_host.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module int, list, process, string.

:- type news
    --->    ended(string)
    ;       note(int).

    % Ends once it has a number.
:- process worker receives int.

worker(Self, !IO) :-
    receive [io(!IO)] (
        select(Self, _) ->
            true
    ).

:- func on_down(down_info) = news.

on_down(Info) = ended(reason_text(reason(Info))).

:- pred monitors(io::di, io::uo) is det.

monitors(!IO) :-
    new_mailbox(Events, !IO),
    spawn(worker, Worker, !IO),
    monitor(Worker, Events, on_down, _, !IO),
    send(Events, note(1), !IO),
    send(Events, note(2), !IO),
    send(Worker, 0, !IO),
    receive [io(!IO)] (
        select(Events, ended(Why)) ->
            io.write_string("down: " ++ Why ++ "\n", !IO)
    ),
    notes(Events, !IO).

:- pred notes(mailbox(news)::in, io::di, io::uo) is det.

notes(Events, !IO) :-
    receive [timeout(after(0)), io(!IO)] (
        select(Events, Event) ->
            io.write_string("then: ", !IO),
            io.write(Event, !IO),
            io.nl(!IO),
            notes(Events, !IO)
    ;
        timeout ->
            true
    ).

:- type entry
    --->    entry(string, int).

    % The values of the key's entries, the last first.
:- pred collect(string::in, mailbox(entry)::in, list(int)::in,
    list(int)::out, io::di, io::uo) is det.

collect(Key, Box, Values0, Values, !IO) :-
    receive [timeout(after(0)), io(!IO)] (
        select(Box, entry(Key, Value)) ->
            collect(Key, Box, [Value | Values0], Values, !IO)
    ;
        timeout ->
            Values = Values0
    ).

:- pred keys(io::di, io::uo) is det.

keys(!IO) :-
    new_mailbox(Box, !IO),
    A = string.from_char_list(['a']),
    B = string.from_char_list(['b']),
    send(Box, entry(B, 1), !IO),
    send(Box, entry(A, 2), !IO),
    send(Box, entry(B, 3), !IO),
    send(Box, entry(A, 4), !IO),
    send(Box, entry(B, 5), !IO),
    send(Box, entry(A, 6), !IO),
    collect(A ++ "", Box, [], As, !IO),
    list.reverse(As, InOrderA),
    io.write_string("key a: ", !IO),
    io.write(InOrderA, !IO),
    io.nl(!IO),
    collect(B ++ "", Box, [], Bs, !IO),
    list.reverse(Bs, InOrderB),
    io.write_string("key b: ", !IO),
    io.write(InOrderB, !IO),
    io.nl(!IO).

:- type size
    --->    small(int)
    ;       large(int).

:- pred sizes(mailbox(size)::in, mailbox(size)::in, list(size)::out,
    io::di, io::uo) is det.

sizes(Small, Large, Sizes, !IO) :-
    receive [timeout(after(0)), io(!IO)] (
        select(Small, small(N)) ->
            sizes(Small, Large, Sizes0, !IO),
            Sizes = [small(N) | Sizes0]
    ;
        select(Large, large(N)) ->
            sizes(Small, Large, Sizes0, !IO),
            Sizes = [large(N) | Sizes0]
    ;
        timeout ->
            Sizes = []
    ).

:- pred aliases(io::di, io::uo) is det.

aliases(!IO) :-
    new_mailbox(Box, !IO),
    send(Box, large(5), !IO),
    send(Box, small(6), !IO),
    send(Box, large(7), !IO),
    sizes(Box, Box, Sizes, !IO),
    io.write_string("aliases: ", !IO),
    io.write(Sizes, !IO),
    io.nl(!IO).

    % Each receive looks past the 999 others to take the next in order.
:- pred last(mailbox(int)::in, int::in, int::out, io::di, io::uo) is det.

last(Box, Next, Last, !IO) :-
    receive [timeout(after(0)), io(!IO)] (
        select(Box, Next) ->
            last(Box, Next - 1, Last, !IO)
    ;
        timeout ->
            Last = Next
    ).

:- pred sends(mailbox(int)::in, int::in, int::in, io::di, io::uo) is det.

sends(Box, I, N, !IO) :-
    ( if I =< N then
        send(Box, I, !IO),
        sends(Box, I + 1, N, !IO)
    else
        true
    ).

main(!IO) :-
    monitors(!IO),
    keys(!IO),
    aliases(!IO),
    new_mailbox(Box, !IO),
    sends(Box, 1, 1000, !IO),
    last(Box, 1000, Left, !IO),
    io.write_string("last: " ++ int_to_string(Left + 1000) ++ "\n", !IO).
