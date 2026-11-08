% The first slice (docs/INPROGRESS.md, The device's side of the first
% slice, step 4): node 1's target region exchanges integers with node 2's
% main over TCP. Node 2 registers two mailboxes and listens. Node 1
% connects, looks them up, and sends node 2 its own mailbox, from the
% host. Then its region, in a procedure that it calls, sends 0 to node 2,
% which answers each N with N + 1, and the region answers each with the
% next, until 200, checking that each comes back in order. The region
% gives the number of rounds, and stops node 2 with -1.
%
% On a GPU (omp-target-two-nodes-gpu.test), the region's sends go over the
% offload plugin's RPC server to node 1's host, which sends them on over
% TCP; node 2's answers arrive at node 1's host, whose mailbox the region
% polls. Here the region runs on the host, and sequentially.
%
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %python %S/Inputs/two-nodes.py %t first-slice | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %python %S/Inputs/two-nodes.py %t.sequential first-slice | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp
%
% OUTPUT:first-slice:
% OUTPUT-NEXT:region: connected
% OUTPUT-NEXT:region: rounds in order: 100
% OUTPUT-NEXT:partner: listening
% OUTPUT-NEXT:partner: the region's mailbox: found
% OUTPUT-NEXT:partner: answered: 100
% OUTPUT-NEXT:partner: stopped

:- module omp_target_two_nodes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process, string.

:- func last = int.
last = 200.

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if Arguments = ["partner"] then
        partner(!IO)
    else if Arguments = ["region", Port] then
        region(Port, !IO)
    else
        io.write_string("usage: partner | region PORT\n", !IO),
        io.set_exit_status(2, !IO)
    ).

%---------------------------------------------------------------------------%
% Node 2, on the CPU.

:- pred partner(io::di, io::uo) is det.
partner(!IO) :-
    new_mailbox(Partner : mailbox(int), !IO),
    new_mailbox(Hello : mailbox(mailbox(int)), !IO),
    register("partner", Partner, _, !IO),
    register("hello", Hello, _, !IO),
    listen("127.0.0.1:0", Listening, !IO),
    (
        Listening = ok(Port),
        io.write_string("listening " ++ int_to_string(Port) ++ "\n", !IO),
        io.flush_output(!IO),
        receive(Hello, any, after(60000), GotRegion, !IO),
        (
            GotRegion = yes(Region),
            io.write_string("the region's mailbox: found\n", !IO),
            answer(Partner, Region, 0, Answered, !IO),
            io.write_string("answered: " ++ int_to_string(Answered) ++ "\n",
                !IO),
            io.write_string("stopped\n", !IO)
        ;
            GotRegion = no,
            io.write_string("the region's mailbox: none\n", !IO)
        )
    ;
        Listening = error(Error),
        io.write_string("listen: " ++ Error ++ "\n", !IO)
    ).

    % Answers each N with N + 1 until -1.
:- pred answer(mailbox(int)::in, mailbox(int)::in, int::in, int::out,
    io::di, io::uo) is det.
answer(Partner, Region, !Answered, !IO) :-
    receive(Partner, any, after(60000), Got, !IO),
    ( if Got = yes(N), N >= 0 then
        send(Region, N + 1, !IO),
        !:Answered = !.Answered + 1,
        answer(Partner, Region, !Answered, !IO)
    else
        true
    ).

%---------------------------------------------------------------------------%
% Node 1, whose target region exchanges with node 2.

:- pred region(string::in, io::di, io::uo) is det.
region(Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(Node),
        io.write_string("connected\n", !IO),
        lookup(Node, "partner", after(10000), FoundPartner, !IO),
        lookup(Node, "hello", after(10000), FoundHello, !IO),
        ( if
            FoundPartner = ok(Partner),
            FoundHello = ok(Hello)
        then
            new_mailbox(Mine : mailbox(int), !IO),
            send(Hello, Mine, !IO),
            %$omp target
            (
                exchange(Partner, Mine, 0, 0, Rounds, !IO),
                send(Partner, -1, !IO)
            ),
            io.write_string("rounds in order: " ++ int_to_string(Rounds) ++
                "\n", !IO)
        else
            io.write_string("lookups failed\n", !IO)
        )
    ;
        Connected = error(Error),
        io.write_string("connect: " ++ Error ++ "\n", !IO)
    ).

    % Sends N, and expects N + 1 back, until last.
:- pred exchange(mailbox(int)::in, mailbox(int)::in, int::in, int::in,
    int::out, io::di, io::uo) is det.
exchange(Partner, Mine, N, !Rounds, !IO) :-
    ( if N >= last then
        true
    else
        send(Partner, N, !IO),
        receive(Mine, any, after(10000), Got, !IO),
        (
            Got = yes(M),
            ( if M = N + 1 then
                !:Rounds = !.Rounds + 1
            else
                true
            ),
            exchange(Partner, Mine, M + 1, !Rounds, !IO)
        ;
            Got = no
        )
    ).
