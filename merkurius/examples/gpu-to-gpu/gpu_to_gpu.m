% One GPU sends an array to another GPU, on another node, and gets an answer.
%
% Each round, GPU A sends an array of numbers. GPU B adds them up and sends the
% sum back. The sends and receives run inside the target regions below; the
% network, the two hosts and the copies in between are the runtime's job.
%
% How the nodes find each other, and what the hosts check, are in demo_node.m.
% Run it with the Makefile in the directory above (README.md).

:- module gpu_to_gpu.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, demo_node, int, io, process.

:- type numbers == array(int).

:- func rounds = int.
rounds = 3.

:- func array_size = int.
array_size = 1000.

main(!IO) :-
    demo_node.start(Node, !IO),
    (
        Node = node_a(Links),
        node_a(Links, 1, !IO)
    ;
        Node = node_b(Links),
        node_b(Links, 1, !IO)
    ;
        Node = not_started
    ).

%---------------------------------------------------------------------------%
% Each node's host: runs its GPU's region once a round.

:- pred node_a(links(numbers, int)::in, int::in, io::di, io::uo) is det.
node_a(Links, Round, !IO) :-
    ( if Round > rounds then
        true
    else
        report_round("A", Round, !IO),
        gpu_a(Links, Round, Sum, !IO),
        check_sum(Sum, array_size * Round, !IO),
        node_a(Links, Round + 1, !IO)
    ).

:- pred node_b(links(numbers, int)::in, int::in, io::di, io::uo) is det.
node_b(Links, Round, !IO) :-
    ( if Round > rounds then
        true
    else
        report_round("B", Round, !IO),
        gpu_b(Links, Size, !IO),
        check_received(Size, !IO),
        node_b(Links, Round + 1, !IO)
    ).

%---------------------------------------------------------------------------%
% How a message gets from one GPU to the other.
%
% The program only sees mailboxes: new_mailbox, send and receive, from
% library/process.m, the same on a GPU and on a CPU. A mailbox lives on the
% host of the node that made it, and a handle to it says which node that is, so
% `send` needs no address and the runtime picks the route.
%
% An array from GPU A to GPU B:
%
%   GPU A          send: a call to its host through LLVM's GPU RPC, in memory
%                  both can reach. The region waits for the answer.
%   host A         The offload plugin's RPC server thread takes the call. It
%                  copies the array from the GPU (omp_target_memcpy), encodes
%                  it, and writes it to the node's TCP socket, all on that
%                  thread. There is no separate forwarding thread.
%   TCP            A plain socket between the two nodes.
%   host B         A reader thread, one for each connection, decodes the bytes
%                  into a message and queues it in the mailbox. If a region is
%                  receiving from that mailbox, it also writes the array's size
%                  to the mailbox's ring, in host memory that the GPU can read.
%   GPU B          receive: polls the ring, with no call to the host. It sees
%                  the size, makes an array, and asks by RPC for the contents,
%                  which host B's RPC server thread copies to the GPU
%                  (omp_target_memcpy).
%
% The sum, a single word, goes the same way without the copies to and from the
% GPU: its value is in the RPC call and then in the ring's slot, so GPU A reads
% it straight from the ring.
%
% One RPC server thread serves all of a node's GPUs, so a send that is
% stalled writing to the socket holds up that node's other GPU calls.

%---------------------------------------------------------------------------%
% What runs on the GPUs.

    % GPU A: makes an array, sends it to GPU B, and waits for the sum.
:- pred gpu_a(links(numbers, int)::in, int::in, int::out, io::di, io::uo)
    is det.
gpu_a(links(ToB, ToA), Round, Sum, !IO) :-
    %$omp target
    (
        Numbers = array.init(array_size, Round),
        say("[GPU A] sending numbers, each", Round, !IO),
        send(ToB, Numbers, !IO),
        receive(ToA, any, after(10000), Answer, !IO),
        (
            Answer = message(Sum),
            say("[GPU A] got the sum", Sum, !IO)
        ;
            Answer = timeout,
            Sum = -1
        )
    ).

    % GPU B: waits for an array, adds it up, and sends the sum to GPU A.
:- pred gpu_b(links(numbers, int)::in, int::out, io::di, io::uo) is det.
gpu_b(links(ToB, ToA), Size, !IO) :-
    %$omp target
    (
        receive(ToB, any, after(10000), Got, !IO),
        (
            Got = message(Numbers),
            Size = array.size(Numbers),
            say("[GPU B] got this many numbers:", Size, !IO),
            Sum = sum_of(Numbers),
            say("[GPU B] sending back their sum", Sum, !IO),
            send(ToA, Sum, !IO)
        ;
            Got = timeout,
            Size = 0
        )
    ).

    % Printed by the host, over the same route as a send.
:- pred say(string::in, int::in, io::di, io::uo) is det.
say(Text, Number, !IO) :-
    io.write_string(Text, !IO),
    io.write_string(" ", !IO),
    io.write_int(Number, !IO),
    io.nl(!IO).

    % Adds up an array with all of the GPU's threads in the region's team.
:- func sum_of(numbers) = int.
sum_of(Numbers) = Sum :-
    Last = array.size(Numbers) - 1,
    %$omp parallel do reduction((+) : Sum)
    ( for(I, 0, Last), fromto(0, S0, S1, Sum) do
        S1 = S0 + Numbers ^ elem(I)
    ).
