% Processes on one worker (design document, "Processes and scheduler"): a
% spawned process receives, replies, waits with a timeout and ends, closing
% its mailbox; a chain of them passes a message along; each has its own
% stack, of the size asked for, and heap. Only its owner receives from a
% mailbox. The program ends when main does.
%
% RUN: merkc -O0 %s -o %t.0 && %t.0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.2 && %t.2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: sed 's/^which = 0\./which = 1./' %s > %t.1.m && merkc %t.1.m -o %t.1
% RUN: not %t.1 2> %t.1.err | FileCheck %s --check-prefix=NOTHING --allow-empty
% RUN: FileCheck %s --check-prefix=OWNER --match-full-lines --strict-whitespace < %t.1.err
% RUN: sed 's/^which = 0\./which = 2./' %s > %t.3.m && merkc %t.3.m -o %t.3
% RUN: not %t.3 2> %t.3.err | FileCheck %s --check-prefix=NOTHING --allow-empty
% RUN: FileCheck %s --check-prefix=OVERFLOW --match-full-lines --strict-whitespace < %t.3.err

:- module process_spawn.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process.

:- func which = int.
which = 0.

:- type request
    --->    ping(int, mailbox(int))
    ;       stop(mailbox(int)).

:- pred server(mailbox(request)::in, io::di, io::uo) is det.
server(Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    ( if Got = yes(ping(N, Reply)) then
        send(Reply, length(1 .. N) * 2, !IO),
        server(Box, !IO)
    else if Got = yes(stop(Reply)) then
        send(Reply, -1, !IO)
    else
        true
    ).

    % Adds one and passes it on.
:- pred link(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
link(Next, Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    ( if Got = yes(N) then send(Next, N + 1, !IO) else true ).

:- pred chain(int::in, mailbox(int)::in, mailbox(int)::out,
    io::di, io::uo) is det.
chain(Count, Last, First, !IO) :-
    ( if Count = 0 then
        First = Last
    else
        spawn(link(Last), Box, !IO),
        chain(Count - 1, Box, First, !IO)
    ).

    % Waits with a timeout, and tells whether a message came.
:- pred timer(mailbox(string)::in, mailbox(int)::in, io::di, io::uo) is det.
timer(Reply, Box, !IO) :-
    receive(Box, any, after(5), Got, !IO),
    (
        Got = no,
        send(Reply, "timed out", !IO)
    ;
        Got = yes(_),
        send(Reply, "got one", !IO)
    ).

:- func deep(int) = int.
deep(N) = (if N = 0 then 0 else (deep(N - 1) * 3 + N) mod 7).

:- pred deep_body(int::in, mailbox(int)::in, mailbox(int)::in,
    io::di, io::uo) is det.
deep_body(Depth, Reply, _, !IO) :-
    send(Reply, deep(Depth), !IO).

:- pred thief(mailbox(request)::in, mailbox(int)::in, io::di, io::uo) is det.
thief(Other, _, !IO) :-
    receive(Other, any, after(0), _, !IO).

main(!IO) :-
    W = which,
    new_mailbox(Reply, !IO),
    ( if W = 1 then
        new_mailbox(Mine : mailbox(request), !IO),
        spawn(thief(Mine), _ : mailbox(int), !IO),
        receive(Reply, any, infinity, _ : maybe(int), !IO)
    else if W = 2 then
        spawn_with_stack(100000, deep_body(100000000, Reply), _, !IO),
        receive(Reply, any, infinity, _ : maybe(int), !IO)
    else
        run(Reply, !IO)
    ).

:- pred run(mailbox(int)::in, io::di, io::uo) is det.
run(Reply, !IO) :-
    spawn(server, Server, !IO),
    send(Server, ping(21, Reply), !IO),
    receive(Reply, any, infinity, Doubled, !IO),
    io.write_line(Doubled, !IO),

    new_mailbox(End, !IO),
    chain(1000, End, First, !IO),
    send(First, 0, !IO),
    receive(End, any, infinity, Passed, !IO),
    io.write_line(Passed, !IO),

    new_mailbox(Told, !IO),
    spawn(timer(Told), _, !IO),
    receive(Told, any, infinity, Timer1, !IO),
    io.write_line(Timer1, !IO),
    spawn(timer(Told), Timer, !IO),
    send(Timer, 1, !IO),
    receive(Told, any, infinity, Timer2, !IO),
    io.write_line(Timer2, !IO),

    % It has a stack far larger than this thread's would be.
    spawn(deep_body(1000000, Reply), _, !IO),
    receive(Reply, any, infinity, Deep, !IO),
    io.write_line(Deep, !IO),

    % After it ends, its mailbox is closed, and a send is dropped.
    send(Server, stop(Reply), !IO),
    receive(Reply, any, infinity, Stopped, !IO),
    io.write_line(Stopped, !IO),
    receive(Reply, any, after(1), Nothing, !IO),
    io.write_line(Nothing, !IO),
    send(Server, ping(1, Reply), !IO),
    receive(Reply, any, after(5), Dropped, !IO),
    io.write_line(Dropped, !IO),

    % The program does not wait for this one.
    new_mailbox(Forever, !IO),
    spawn(link(Forever), _, !IO),
    io.write_string("done\n", !IO).

% OUTPUT:yes(42)
% OUTPUT-NEXT:yes(1000)
% OUTPUT-NEXT:yes("timed out")
% OUTPUT-NEXT:yes("got one")
% OUTPUT-NEXT:yes({{[0-6]}})
% OUTPUT-NEXT:yes(-1)
% OUTPUT-NEXT:no
% OUTPUT-NEXT:no
% OUTPUT-NEXT:done
% OUTPUT-NOT:{{.}}

% NOTHING-NOT:{{.}}
% OWNER:Mercury runtime: a process received from a mailbox of another process
% OVERFLOW:Mercury runtime: stack overflow
