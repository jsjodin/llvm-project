% Receive goals, receive [Params] ( Arms ), as the converter reads them with
% -freceive-goal (design document, "Processes and mailboxes"): each becomes
% a ReceiveGoal with its parameters and its arms, in order. receive(...)
% written as a call stays a call. Without -freceive-goal, receive is not an
% operator, as in mmc, and the goals are syntax errors.
%
% RUN: merkc -fc1 -freceive-goal -fdebug-dump-parse-tree-no-sema %s \
% RUN:   | FileCheck %s
% RUN: not merkc -fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --check-prefix=PLAIN
% PLAIN: receive.m:{{[0-9]+}}:{{[0-9]+}}: error: expected an operator or '.', but found '['
% Semantics/desugar/receive.m desugars them.

:- module receive_goals.
:- interface.
:- import_module io, process.

:- type counter_msg
    --->    increment(int)
    ;       get(mailbox(int))
    ;       stop.

:- pred loop(int::in, mailbox(counter_msg)::in, mailbox(string)::in,
    io::di, io::uo) is det.

:- implementation.
:- import_module int, maybe.

    % The design document's loop: arms for two mailboxes and a timeout.
loop(Count, Self, Downs, !IO) :-
    receive [timeout(after(5000)), io(!IO)] (
        select(Self, increment(N)) ->
            loop(Count + N, Self, Downs, !IO)
    ;
        select(Self, get(ReplyTo)) ->
            send(ReplyTo, Count, !IO),
            loop(Count, Self, Downs, !IO)
    ;
        select(Self, stop) ->
            true
    ;
        select(Downs, _) ->
            io.write_string("worker died\n", !IO),
            loop(Count, Self, Downs, !IO)
    ;
        timeout ->
            io.write_string("idle\n", !IO),
            loop(Count, Self, Downs, !IO)
    ).
% CHECK:      | | | name: SymName loop
% CHECK:      | | | body: ReceiveGoal
% CHECK-NEXT: | | | | params:
% CHECK-NEXT: | | | | | ReceiveParam
% CHECK-NEXT: | | | | | | kind: Timeout
% CHECK-NEXT: | | | | | | value: FunctorData
% CHECK-NEXT: | | | | | | | name: SymName after
% CHECK-NEXT: | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | Literal 5000
% CHECK-NEXT: | | | | | ReceiveParam
% CHECK-NEXT: | | | | | | kind: Io
% CHECK-NEXT: | | | | | | value: StateVar !IO
% CHECK-NEXT: | | | | arms:
% CHECK-NEXT: | | | | | ReceiveArm
% CHECK-NEXT: | | | | | | mailbox: Variable Self
% CHECK-NEXT: | | | | | | pattern: FunctorData
% CHECK-NEXT: | | | | | | | name: SymName increment
% CHECK-NEXT: | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | Variable N
% CHECK-NEXT: | | | | | | goal: Call
% CHECK:      | | | | | ReceiveArm
% CHECK-NEXT: | | | | | | mailbox: Variable Self
% CHECK-NEXT: | | | | | | pattern: FunctorData
% CHECK-NEXT: | | | | | | | name: SymName get
% CHECK:      | | | | | | goal: Conjunction
% CHECK:      | | | | | ReceiveArm
% CHECK-NEXT: | | | | | | mailbox: Variable Self
% CHECK-NEXT: | | | | | | pattern: FunctorData
% CHECK-NEXT: | | | | | | | name: SymName stop
% CHECK-NEXT: | | | | | | | arguments:
% CHECK-NEXT: | | | | | | goal: BuiltinGoal
% CHECK:      | | | | | ReceiveArm
% CHECK-NEXT: | | | | | | mailbox: Variable Downs
% CHECK-NEXT: | | | | | | pattern: Variable _
% CHECK-NEXT: | | | | | | goal: Conjunction
% CHECK:      | | | | | ReceiveArm
% CHECK-NEXT: | | | | | | mailbox: none
% CHECK-NEXT: | | | | | | pattern: none
% CHECK-NEXT: | | | | | | goal: Conjunction

    % One arm, without a timeout; a receive nested in an arm; and receive
    % written as a call, qualified or not, which is the library's predicate.
:- pred reply(mailbox(int)::in, mailbox(mailbox(int))::in, int::out,
    io::di, io::uo) is det.
reply(Replies, Requests, Got, !IO) :-
    receive [io(!IO)] ( select(Requests, R) -> send(R, 1, !IO) ),
    receive [io(!IO), timeout(after(0))] (
        timeout ->
            Got = 0
    ;
        select(Replies, N) ->
            process.receive(Replies, any, after(0), _, !IO),
            receive(Replies, any, after(0), _, !IO),
            receive [io(!IO)] ( select(Replies, M) -> Got = N + M )
    ).
% CHECK:      | | | name: SymName reply
% CHECK:      | | | | | ReceiveGoal
% CHECK-NEXT: | | | | | | params:
% CHECK-NEXT: | | | | | | | ReceiveParam
% CHECK-NEXT: | | | | | | | | kind: Io
% CHECK:      | | | | | | arms:
% CHECK-NEXT: | | | | | | | ReceiveArm
% CHECK-NEXT: | | | | | | | | mailbox: Variable Requests
% CHECK-NEXT: | | | | | | | | pattern: Variable R
% CHECK-NEXT: | | | | | | | | goal: Call
% CHECK:      | | | | | ReceiveGoal
% CHECK:      | | | | | | arms:
% CHECK-NEXT: | | | | | | | ReceiveArm
% CHECK-NEXT: | | | | | | | | mailbox: none
% CHECK-NEXT: | | | | | | | | pattern: none
% CHECK-NEXT: | | | | | | | | goal: Unification
% CHECK:      | | | | | | | ReceiveArm
% CHECK-NEXT: | | | | | | | | mailbox: Variable Replies
% CHECK-NEXT: | | | | | | | | pattern: Variable N
% CHECK-NEXT: | | | | | | | | goal: Conjunction
% CHECK-NEXT: | | | | | | | | | parallel: false
% CHECK-NEXT: | | | | | | | | | goals:
% CHECK-NEXT: | | | | | | | | | | Call
% CHECK-NEXT: | | | | | | | | | | | name: SymName process.receive
% CHECK:      | | | | | | | | | | Call
% CHECK-NEXT: | | | | | | | | | | | name: SymName receive
% CHECK:      | | | | | | | | | | ReceiveGoal
