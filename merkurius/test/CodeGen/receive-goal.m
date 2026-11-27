% Receive goals, with -freceive-goal (design document, "The receive goal"),
% which desugaring makes a receive over a selector of the arms and a switch
% on the result:
%
% - The design document's counter loop, with arms for two mailboxes and a
%   timeout, which threads !IO and a count through its arms.
% - The oldest message that some arm selects is taken, across mailboxes,
%   whatever the order of the arms; a mailbox's arms, wherever they are
%   written, are tried in order, so a message that two select goes to the
%   first.
% - A message that no arm selects stays queued.
% - A pattern's variable that occurs outside the goal is tested, as Ref is;
%   the others are the arm's own, none, one or several.
% - A mailbox given by an expression, and a receive nested in an arm.
% - A timeout arm, and without one, the error when every mailbox is closed,
%   which ends the program.
%
% Each receive goal is a merk.receive, which names each mailbox's matcher
% (docs/INPROGRESS.md, Concurrency in merk, step 1): the counter loop's
% has two cases, Self's and Downs', first written first, and no selector.
% Lowering makes it a loop over the runtime's candidates (Lowering.cpp,
% LowerSelect): a case for each mailbox, then for each candidate a direct
% call of its case's matcher, with no closure, and a take or a skip. The
% matchers' calls are safepoints whose stack maps list what is live at the
% receive, as Self and Downs are; the library's calls make what a monitor
% left a message, and report every mailbox closed.
%
% mmc has no receive goal, so the output is written by hand.
%
% RUN: merkc -freceive-goal -O0 %s -o %t0 && not %t0 > %t0.out 2> %t0.err
% RUN: FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace < %t0.out
% RUN: FileCheck %s --check-prefix=ERROR < %t0.err
% RUN: merkc -freceive-goal -O2 %s -o %t2 && not %t2 > %t2.out 2> %t2.err
% RUN: FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace < %t2.out
% RUN: FileCheck %s --check-prefix=ERROR < %t2.err
% RUN: merkc -fc1 -freceive-goal -emit-merk %s -o - | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -freceive-goal -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM --implicit-check-not='call {{.*}}@process.{{selecting_matching|new_selector|receive_goal}}'
%
% OUTPUT:worker died: worker 1
% OUTPUT-NEXT:idle
% OUTPUT-NEXT:stopped at 12
% OUTPUT-NEXT:counter: 12
% OUTPUT-NEXT:oldest: [b(1), a(2), b(3), a(4)]
% OUTPUT-NEXT:first arm: small(3)
% OUTPUT-NEXT:second arm: large(30)
% OUTPUT-NEXT:selected: 5
% OUTPUT-NEXT:left queued: 7
% OUTPUT-NEXT:reply to 2: 20
% OUTPUT-NEXT:reply to 1: 10
% OUTPUT-NEXT:pair: 3 and "three"
% OUTPUT-NEXT:nothing: stop
% OUTPUT-NEXT:from the record's mailbox: 5
% OUTPUT-NEXT:nested: 1 then 2
% OUTPUT-NEXT:timeout: none
% OUTPUT-NOT:{{.}}
% ERROR: receive: every mailbox is closed
%
% MERK-LABEL: merk.proc private @receive_goal.loop_5_0(
% MERK-NOT: merk.closure
% MERK-NOT: merk.call @process.{{selecting_matching|new_selector|receive_goal}}
% MERK: merk.receive {{.*}} matchers [@receive_goal.loop_5_0$lambda0, @receive_goal.loop_5_0$lambda1] counts [0, 0]
% MERK-SAME: on(%arg1, %arg2 : !merk.term<"process.mailbox(receive_goal.counter_msg)">, !merk.term<"process.mailbox(string)">)
% MERK-SAME: -> !merk.term<"process.receive_result(process.receive_arms(int,
% MERK-NOT: merk.closure
% MERK-NOT: merk.call @process.{{selecting_matching|new_selector|receive_goal}}
% MERK-LABEL: merk.proc private @receive_goal.loop_5_0$lambda0(
% MERK-SAME: !merk.term<"receive_goal.counter_msg">) -> !merk.word
% MERK-SAME: determinism = #merk.determinism<semidet>
%
% LLVM-LABEL: define {{.*}} @receive_goal.loop_5_0(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2)
% LLVM: %[[S:[0-9]+]] = call i64 @merk_select_begin(
% LLVM-COUNT-2: call void @merk_select_case(i64 %[[S]], ptr
% LLVM: call i64 @merk_select_next(i64 %[[S]])
% LLVM: call tailcc void @process.select_closed_0_0()
% LLVM: call tailcc ptr addrspace(1) @process.select_resolve_4_0({{.*}}, i64 %[[S]])
% LLVM: call tailcc { i1, ptr addrspace(1) } @"receive_goal.loop_5_0$lambda0$select"({{.*}}) [ "deopt"(i64 {{[0-9]+}}, ptr addrspace(1) %1, i64 {{[0-9]+}}, ptr addrspace(1) %2,
% LLVM: call void @merk_select_take(i64 %[[S]])
% LLVM: call void @merk_select_skip(i64 %[[S]])
% LLVM: call tailcc { i1, ptr addrspace(1) } @"receive_goal.loop_5_0$lambda1$select"({{.*}}) [ "deopt"(i64 {{[0-9]+}}, ptr addrspace(1) %1, i64 {{[0-9]+}}, ptr addrspace(1) %2,
% LLVM-LABEL: define {{.*}} @"receive_goal.loop_5_0$lambda0$select"(
% LLVM: call {{.*}} @"receive_goal.loop_5_0$lambda0"(

:- module receive_goal.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module int, list, process, string.

:- type counter_msg
    --->    increment(int)
    ;       get(mailbox(int))
    ;       stop.

    % The design document's loop.
:- pred loop(int::in, mailbox(counter_msg)::in, mailbox(string)::in,
    io::di, io::uo) is det.
loop(Count, Self, Downs, !IO) :-
    receive [timeout(after(50)), io(!IO)] (
        select(Self, increment(N)) ->
            loop(Count + N, Self, Downs, !IO)
    ;
        select(Self, get(ReplyTo)) ->
            send(ReplyTo, Count, !IO),
            loop(Count, Self, Downs, !IO)
    ;
        select(Self, stop) ->
            io.write_string("stopped at " ++ int_to_string(Count) ++ "\n", !IO)
    ;
        select(Downs, Why) ->
            io.write_string("worker died: " ++ Why ++ "\n", !IO),
            loop(Count, Self, Downs, !IO)
    ;
        timeout ->
            io.write_string("idle\n", !IO),
            send(Self, stop, !IO),
            loop(Count, Self, Downs, !IO)
    ).

:- pred counter(io::di, io::uo) is det.
counter(!IO) :-
    new_mailbox(Self, !IO),
    new_mailbox(Downs, !IO),
    new_mailbox(Replies, !IO),
    send(Self, increment(5), !IO),
    send(Self, increment(7), !IO),
    send(Self, get(Replies), !IO),
    send(Downs, "worker 1", !IO),
    % The loop runs until the get, the down and the timeout's stop are
    % handled; the reply waits in Replies.
    loop(0, Self, Downs, !IO),
    receive [io(!IO)] (
        select(Replies, Count) ->
            io.write_string("counter: " ++ int_to_string(Count) ++ "\n", !IO)
    ).

:- type letter
    --->    a(int)
    ;       b(int).

:- type size
    --->    small(int)
    ;       large(int).

    % Receives from either mailbox until both are empty.
:- pred oldest(mailbox(int)::in, mailbox(int)::in, list(letter)::out,
    io::di, io::uo) is det.
oldest(A, B, Letters, !IO) :-
    receive [timeout(after(0)), io(!IO)] (
        select(B, N) ->
            oldest(A, B, Letters0, !IO),
            Letters = [b(N) | Letters0]
    ;
        select(A, N) ->
            oldest(A, B, Letters0, !IO),
            Letters = [a(N) | Letters0]
    ;
        timeout ->
            Letters = []
    ).

:- pred sizes(io::di, io::uo) is det.
sizes(!IO) :-
    new_mailbox(A, !IO),
    new_mailbox(B, !IO),
    send(B, 1, !IO),
    send(A, 2, !IO),
    send(B, 3, !IO),
    send(A, 4, !IO),
    oldest(A, B, Letters, !IO),
    io.write_string("oldest: ", !IO),
    io.write(Letters, !IO),
    io.nl(!IO),
    % Two arms for one mailbox, written apart: 3 goes to the first, though
    % the last takes any number.
    new_mailbox(Numbers, !IO),
    new_mailbox(Words, !IO),
    send(Numbers, 3, !IO),
    take_size(Numbers, Words, First, !IO),
    io.write_string("first arm: ", !IO),
    io.write(First, !IO),
    io.nl(!IO),
    send(Numbers, 30, !IO),
    take_size(Numbers, Words, Second, !IO),
    io.write_string("second arm: ", !IO),
    io.write(Second, !IO),
    io.nl(!IO),
    % No arm selects 7, which stays queued.
    send(Numbers, 7, !IO),
    send(Numbers, 5, !IO),
    receive [io(!IO)] (
        select(Numbers, 5) ->
            io.write_string("selected: 5\n", !IO)
    ),
    receive [io(!IO)] (
        select(Numbers, Left) ->
            io.write_string("left queued: " ++ int_to_string(Left) ++ "\n",
                !IO)
    ).

:- pred take_size(mailbox(int)::in, mailbox(string)::in, size::out,
    io::di, io::uo) is det.
take_size(Numbers, Words, Size, !IO) :-
    receive [io(!IO)] (
        select(Numbers, 3) ->
            Size = small(3)
    ;
        select(Words, _) ->
            Size = small(0)
    ;
        select(Numbers, N) ->
            Size = large(N)
    ).

:- type reply
    --->    reply(int, int).

    % Ref occurs outside the goal, so the pattern tests it: the replies are
    % taken in the order asked for, not the order they arrived in.
:- pred ask(mailbox(reply)::in, int::in, io::di, io::uo) is det.
ask(Replies, Ref, !IO) :-
    receive [io(!IO)] (
        select(Replies, reply(Ref, Value)) ->
            io.write_string("reply to " ++ int_to_string(Ref) ++ ": " ++
                int_to_string(Value) ++ "\n", !IO)
    ).

:- type mixed
    --->    pair(int, string)
    ;       stop.

:- type record
    --->    record(inbox :: mailbox(int)).

:- pred variables(io::di, io::uo) is det.
variables(!IO) :-
    new_mailbox(Replies, !IO),
    send(Replies, reply(1, 10), !IO),
    send(Replies, reply(2, 20), !IO),
    ask(Replies, 2, !IO),
    ask(Replies, 1, !IO),
    new_mailbox(Mixed, !IO),
    send(Mixed, pair(3, "three"), !IO),
    send(Mixed, stop, !IO),
    % Several variables, then none.
    receive [io(!IO)] (
        select(Mixed, pair(N, S)) ->
            io.write_string("pair: " ++ int_to_string(N) ++ " and \"" ++ S ++
                "\"\n", !IO)
    ;
        select(Mixed, stop) ->
            io.write_string("nothing: stop\n", !IO)
    ),
    receive [io(!IO)] (
        select(Mixed, pair(_, _)) ->
            io.write_string("pair again\n", !IO)
    ;
        select(Mixed, stop) ->
            io.write_string("nothing: stop\n", !IO)
    ),
    % A mailbox given by an expression.
    new_mailbox(Inbox, !IO),
    R = record(Inbox),
    send(R ^ inbox, 5, !IO),
    receive [io(!IO)] (
        select(R ^ inbox, Five) ->
            io.write_string("from the record's mailbox: " ++
                int_to_string(Five) ++ "\n", !IO)
    ),
    % A receive nested in an arm.
    new_mailbox(Outer, !IO),
    new_mailbox(Inner, !IO),
    send(Outer, 1, !IO),
    send(Inner, 2, !IO),
    receive [io(!IO)] (
        select(Outer, X) ->
            receive [io(!IO)] (
                select(Inner, Y) ->
                    io.write_string("nested: " ++ int_to_string(X) ++
                        " then " ++ int_to_string(Y) ++ "\n", !IO)
            )
    ).

:- pred timeouts(io::di, io::uo) is det.
timeouts(!IO) :-
    new_mailbox(Empty, !IO),
    receive [timeout(after(20)), io(!IO)] (
        select(Empty, N) ->
            io.write_string("timeout: got " ++ int_to_string(N) ++ "\n", !IO)
    ;
        timeout ->
            io.write_string("timeout: none\n", !IO)
    ),
    % Without a timeout arm, a receive whose mailboxes are all closed is an
    % error.
    close_mailbox(Empty, !IO),
    receive [io(!IO)] (
        select(Empty, M) ->
            io.write_string("closed: got " ++ int_to_string(M) ++ "\n", !IO)
    ).

main(!IO) :-
    counter(!IO),
    sizes(!IO),
    variables(!IO),
    timeouts(!IO).
