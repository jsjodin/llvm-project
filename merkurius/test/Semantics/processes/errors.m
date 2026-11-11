% Errors in process declarations after conversion (design document,
% "Processes and mailboxes"): the declarations are ordinary ones, so a kind
% declared twice, or with its predicate declared too, is a predicate declared
% more than once, reported for the kind's predicate alone, as the spawn
% predicates' names are the compiler's own; a predicate of the program named
% spawn_Kind is the program's, which no kind's spawn predicate clashes with.
% A spawn of a name that is no kind's, written out, is an error. What is not
% implemented yet is reported: several message
% types, and spawn_Kind_on, which needs remote spawn; a spawn_Name_on that is
% no kind's stays an undefined predicate. A kind that runs on a GPU is an
% ordinary kind here, with no error.
%
% RUN: not merkc -fc1 -fprocess-declarations -fdebug-dump-resolved-names %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module errors.
:- interface.
:- import_module io, process.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type msg ---> m.

:- process counter(int) receives msg.
:- pred counter(int::in, mailbox(msg)::in, io::di, io::uo) is det.
% CHECK: errors.m:[[@LINE-1]]:9: error: pred errors.counter/4 is declared more than once
% CHECK: errors.m:[[@LINE-3]]:4: because: it is first declared here
counter(_, _, !IO).

:- process counter(int) receives msg.
% CHECK: errors.m:[[@LINE-1]]:4: error: pred errors.counter/4 is declared more than once
% CHECK: errors.m:[[@LINE-8]]:4: because: it is first declared here

:- pred spawn_counter(int::in, mailbox(msg)::out, io::di, io::uo) is det.
spawn_counter(_, M, !IO) :-
    new_mailbox(M, !IO).

:- process server(int) receives msg, string.
% CHECK: errors.m:[[@LINE-1]]:4: error: not yet implemented: process kinds that receive several message types
server(_, _, _, !IO).

:- process summer(int) receives msg on gpu.
summer(_, _, !IO).

:- pred spawn_thing(int::in, io::di, io::uo) is det.
spawn_thing(_, !IO).

main(!IO) :-
    spawn_counter(1, _, !IO),
    spawn(host, thing(1), _, !IO),
% CHECK: errors.m:[[@LINE-1]]:17: error: spawn's second argument must be a process kind applied to its arguments, but thing/1 is not a process kind that receives one message type
    spawn(host, server(1), _, !IO),
% CHECK: errors.m:[[@LINE-1]]:17: error: spawn's second argument must be a process kind applied to its arguments, but server/1 is not a process kind that receives one message type
    spawn_counter_on(1, 2, _, !IO),
% CHECK: errors.m:[[@LINE-1]]:5: error: not yet implemented: spawning a process kind on another node (spawn_counter_on/5)
    errors.spawn_counter_on(1, 2, _, !IO),
% CHECK: errors.m:[[@LINE-1]]:5: error: not yet implemented: spawning a process kind on another node (errors.spawn_counter_on/5)
    spawn_counter_on(1, 2, !IO),
% CHECK: errors.m:[[@LINE-1]]:5: error: undefined predicate spawn_counter_on/4
    io.spawn_counter_on(1, 2, _, !IO),
% CHECK: errors.m:[[@LINE-1]]:5: error: undefined predicate io.spawn_counter_on/5
    spawn_thing_on(1, 2, !IO).
% CHECK: errors.m:[[@LINE-1]]:5: error: undefined predicate spawn_thing_on/4
