% A spawned process's closure, and a monitor's function for another process's
% mailbox, are copies: the captured list is at another address (design
% document, Closures that say what they captured).
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_spawn_copies.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process.

:- func address(list(int)) = int.
:- pragma foreign_proc("C",
    address(L::in) = (A::out),
    [will_not_call_mercury, promise_pure, thread_safe],
    "A = (MR_Integer) L;").

:- func make(int) = list(int).
make(N) = ( if N = 0 then [] else [N | make(N - 1)] ).

:- pred child(list(int)::in, mailbox(int)::in, mailbox(int)::in,
    io::di, io::uo) is det.
child(L, Out, _, !IO) :-
    send(Out, address(L), !IO),
    send(Out, list.foldl(plus, L, 0), !IO),
    % A monitor of itself, whose function main applies.
    new_mailbox(Mine : mailbox(int), !IO),
    Mine2 = make(4),
    send(Out, address(Mine2), !IO),
    monitor(Mine, Out, (func(_) = address(Mine2)), _, !IO).

:- pred take(mailbox(int)::in, int::out, io::di, io::uo) is det.
take(Out, V, !IO) :-
    receive(Out, any, after(20000), Got, !IO),
    (
        Got = message(V)
    ;
        Got = timeout,
        V = -1
    ).

:- pred say(bool_name::in, io::di, io::uo) is det.
say(differ, !IO) :- io.write_string("differ\n", !IO).
say(same, !IO) :- io.write_string("same\n", !IO).

:- type bool_name ---> differ ; same.

:- func compare_addresses(int, int) = bool_name.
compare_addresses(A, B) = ( if A = B then same else differ ).

main(!IO) :-
    L = make(3),
    new_mailbox(Out, !IO),
    spawn(child(L, Out), _, !IO),
    take(Out, InChild, !IO),
    say(compare_addresses(address(L), InChild), !IO),
    take(Out, Sum, !IO),
    io.write_int(Sum, !IO),
    io.nl(!IO),
    take(Out, Theirs, !IO),
    take(Out, Applied, !IO),
    say(compare_addresses(Theirs, Applied), !IO).

% OUTPUT:differ
% OUTPUT-NEXT:6
% OUTPUT-NEXT:differ
