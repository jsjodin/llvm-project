% Process declarations, with -fprocess-declarations (design document,
% "Processes and mailboxes"): spawn_Kind spawns a process that runs the
% kind's predicate with the arguments and the process's mailbox.
%
% - A kind of this module, counter, spawned twice, so that each process has
%   its own count.
% - Kinds of another module's interface: echo, polymorphic in its messages,
%   spawned at two types, and logger, which has no arguments, spawned
%   unqualified and qualified.
%
% mmc has no process declarations, so the output is written by hand.
%
% RUN: merkc -O0 -fprocess-declarations -I %S/Inputs %S/Inputs/process_kinds.m %s -o %t0 && %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations -I %S/Inputs %S/Inputs/process_kinds.m %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module process_declarations.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, process_kinds, string.

:- type counter_msg
    --->    add(int)
    ;       get(mailbox(int))
    ;       stop.

:- process counter(int) receives counter_msg.

counter(N, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(add(K)) then
        counter(N + K, Self, !IO)
    else if R = message(get(Reply)) then
        send(Reply, N, !IO),
        counter(N, Self, !IO)
    else
        true
    ).

:- pred get(mailbox(counter_msg)::in, mailbox(int)::in, io::di, io::uo)
    is det.
get(Counter, Reply, !IO) :-
    send(Counter, get(Reply), !IO),
    receive(Reply, any, infinity, R, !IO),
    ( if R = message(N) then
        io.write_string("count " ++ int_to_string(N) ++ "\n", !IO)
    else
        io.write_string("timeout\n", !IO)
    ).

main(!IO) :-
    new_mailbox(Reply, !IO),
    spawn_counter(10, C1, !IO),
    spawn_counter(100, C2, !IO),
    send(C1, add(5), !IO),
    send(C2, add(1), !IO),
    send(C1, add(2), !IO),
    get(C1, Reply, !IO),
    get(C2, Reply, !IO),
    send(C1, stop, !IO),
    send(C2, stop, !IO),

    spawn_echo(0, E1, !IO),
    new_mailbox(Strings, !IO),
    send(E1, echo("hello", Strings), !IO),
    receive(Strings, any, infinity, R1, !IO),
    spawn_echo(0, E2, !IO),
    new_mailbox(Lists, !IO),
    send(E2, echo([1, 2, 3], Lists), !IO),
    receive(Lists, any, infinity, R2, !IO),
    send(E1, quit, !IO),
    send(E2, quit, !IO),
    ( if R1 = message(S), R2 = message(L) then
        io.write_string(S ++ "\n", !IO),
        io.write(L, !IO),
        io.nl(!IO)
    else
        io.write_string("timeout\n", !IO)
    ),

    new_mailbox(Counts, !IO),
    spawn_logger(Log1, !IO),
    send(Log1, line("one"), !IO),
    send(Log1, line("two"), !IO),
    send(Log1, done(Counts), !IO),
    receive(Counts, any, infinity, C3, !IO),
    io.write(C3, !IO),
    io.nl(!IO),
    process_kinds.spawn_logger(Log2, !IO),
    send(Log2, line("three"), !IO),
    send(Log2, done(Counts), !IO),
    receive(Counts, any, infinity, C4, !IO),
    io.write(C4, !IO),
    io.nl(!IO).

% CHECK:count 17
% CHECK-NEXT:count 101
% CHECK-NEXT:hello
% CHECK-NEXT:[1, 2, 3]
% CHECK-NEXT:log: one
% CHECK-NEXT:log: two
% CHECK-NEXT:message(2)
% CHECK-NEXT:log: three
% CHECK-NEXT:message(1)
