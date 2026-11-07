% Receive goals, with -freceive-goal (design document, "The receive goal"):
% a receive over a selector with a case for each mailbox, the first written
% added first, whose predicate tries that mailbox's arms in order, and a
% switch on the result. A pattern's variables that occur outside the goal,
% as Ref does, are tested in the predicate; the others are fresh copies
% there, which come back through the result, receive_rest applied to the
% arm's number of receive_arm, or for the last arm only receive_rest, and
% with several variables, a tuple. Each goal's nonlocal variables are after
% "~". CodeGen/receive-goal.m runs receive goals.
%
% RUN: merkc -fc1 -freceive-goal -fdebug-dump-nonlocals %s 2>&1 | FileCheck %s \
% RUN:   --implicit-check-not=warning: --implicit-check-not=error:

:- module receive_goals.
:- interface.
:- import_module io, process.

:- type msg
    --->    reply(int, string)
    ;       stop.

:- pred serve(int::in, mailbox(msg)::in, mailbox(string)::in,
    io::di, io::uo) is det.

:- implementation.

serve(Ref, Self, Downs, !IO) :-
    receive [timeout(after(10)), io(!IO)] (
        select(Self, reply(Ref, S)) ->
            io.write_string(S, !IO)
    ;
        select(Downs, Why) ->
            io.write_string(Why, !IO)
    ;
        select(Self, stop) ->
            true
    ;
        timeout ->
            true
    ).
% CHECK:      Predicate serve(Ref, Self, Downs, IO!0, IO!6)
% CHECK-NEXT: | Conj #48 ~ [IO!0, Ref, Self, Downs, IO!6]
% CHECK-NEXT: | | Unify #12 ReceiveSelector = process.selecting_matching(Self, $6, $11) ~ [Self, ReceiveSelector, $6, $11]
% CHECK-NEXT: | | Unify #10 $6 = pred(Message::in, Selected::out) is semidet ~ [Ref, $6]
% CHECK-NEXT: | | | IfThenElse #9 ~ [Ref, Message, Selected]
% CHECK-NEXT: | | | | Unify #0 Message = reply(Ref, $5) ~ [Ref, $5, Message]
% CHECK-NEXT: | | | | Unify #1 Selected = process.receive_arm($5) ~ [$5, Selected]
% CHECK-NEXT: | | | | IfThenElse #8 ~ [Message, Selected]
% CHECK-NEXT: | | | | | Unify #2 Message = stop ~ [Message]
% CHECK-NEXT: | | | | | Conj #7 ~ [Selected]
% CHECK-NEXT: | | | | | | Unify #5 Selected = process.receive_rest($9) ~ [Selected, $9]
% CHECK-NEXT: | | | | | | Unify #4 $9 = process.receive_rest($10) ~ [$9, $10]
% CHECK-NEXT: | | | | | | Unify #3 $10 = process.receive_none ~ [$10]
% CHECK-NEXT: | | | | | Disj #6 ~ []
% CHECK-NEXT: | | Unify #11 $11 = process.new_selector ~ [$11]
% CHECK-NEXT: | | Unify #20 ReceiveSelector$12 = process.selecting_matching(Downs, $14, ReceiveSelector) ~ [Downs, ReceiveSelector, ReceiveSelector$12, $14]
% CHECK-NEXT: | | Unify #19 $14 = pred(Message$15::in, Selected$16::out) is semidet ~ [$14]
% CHECK-NEXT: | | | IfThenElse #18 ~ [Message$15, Selected$16]
% CHECK-NEXT: | | | | Unify #13 Message$15 = $13 ~ [$13, Message$15]
% CHECK-NEXT: | | | | Conj #17 ~ [$13, Selected$16]
% CHECK-NEXT: | | | | | Unify #15 Selected$16 = process.receive_rest($17) ~ [Selected$16, $17]
% CHECK-NEXT: | | | | | Unify #14 $17 = process.receive_arm($13) ~ [$13, $17]
% CHECK-NEXT: | | | | Disj #16 ~ []
% CHECK-NEXT: | | Unify #22 $19 = after($20) ~ [$19, $20]
% CHECK-NEXT: | | Unify #21 $20 = 10 ~ [$20]
% CHECK-NEXT: | | Call #23 process.receive_goal(ReceiveSelector$12, $19, Received, IO!0, IO!1) ~ [IO!0, ReceiveSelector$12, Received, $19, IO!1]
% CHECK-NEXT: | | Disj #47 ~ [Received, IO!1, IO!6]
% CHECK-NEXT: | | | Conj #45 ~ [Received, IO!1, IO!6]
% CHECK-NEXT: | | | | Unify #24 Received = process.receive_nothing ~ [Received]
% CHECK-NEXT: | | | | Unify #43 IO!6 = IO!1 ~ [IO!1, IO!6]
% CHECK-NEXT: | | | Conj #46 ~ [Received, IO!1, IO!6]
% CHECK-NEXT: | | | | Unify #25 Received = process.receive_message(ReceiveTaken) ~ [Received, ReceiveTaken]
% CHECK-NEXT: | | | | Disj #42 ~ [IO!1, ReceiveTaken, IO!5]
% CHECK-NEXT: | | | | | Conj #40 ~ [IO!1, ReceiveTaken, IO!5]
% CHECK-NEXT: | | | | | | Unify #26 ReceiveTaken = process.receive_arm(S) ~ [ReceiveTaken, S]
% CHECK-NEXT: | | | | | | Call #27 io.write_string(S, IO!1, IO!2) ~ [IO!1, S, IO!2]
% CHECK-NEXT: | | | | | | Unify #38 IO!5 = IO!2 ~ [IO!2, IO!5]
% CHECK-NEXT: | | | | | Conj #41 ~ [IO!1, ReceiveTaken, IO!5]
% CHECK-NEXT: | | | | | | Unify #28 ReceiveTaken = process.receive_rest(ReceiveRest) ~ [ReceiveTaken, ReceiveRest]
% CHECK-NEXT: | | | | | | Disj #37 ~ [IO!1, ReceiveRest, IO!4]
% CHECK-NEXT: | | | | | | | Conj #35 ~ [IO!1, ReceiveRest, IO!4]
% CHECK-NEXT: | | | | | | | | Unify #29 ReceiveRest = process.receive_arm(Why) ~ [ReceiveRest, Why]
% CHECK-NEXT: | | | | | | | | Call #30 io.write_string(Why, IO!1, IO!3) ~ [IO!1, Why, IO!3]
% CHECK-NEXT: | | | | | | | | Unify #33 IO!4 = IO!3 ~ [IO!3, IO!4]
% CHECK-NEXT: | | | | | | | Conj #36 ~ [IO!1, ReceiveRest, IO!4]
% CHECK-NEXT: | | | | | | | | Unify #31 ReceiveRest = process.receive_rest(ReceiveRest$24) ~ [ReceiveRest, ReceiveRest$24]
% CHECK-NEXT: | | | | | | | | Unify #32 ReceiveRest$24 = process.receive_none ~ [ReceiveRest$24]
% CHECK-NEXT: | | | | | | | | Unify #34 IO!4 = IO!1 ~ [IO!1, IO!4]
% CHECK-NEXT: | | | | | | Unify #39 IO!5 = IO!4 ~ [IO!4, IO!5]
% CHECK-NEXT: | | | | Unify #44 IO!6 = IO!5 ~ [IO!5, IO!6]
