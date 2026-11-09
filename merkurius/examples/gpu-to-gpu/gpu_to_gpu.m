% One GPU sends an array to another GPU, on another node, and gets an answer.
%
% Each round, GPU A sends an array of numbers. GPU B adds them up and sends the
% sum back. The sends and receives run inside the target regions below; the
% network, the two hosts and the copies in between are the runtime's job.
%
% How the nodes find each other, and the messages printed, are in demo_node.m.
% Run it with the Makefile in the directory above (README.md).

:- module gpu_to_gpu.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, demo_node, int, maybe, process.

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
% Each node's host: runs its GPU's region once a round, and reports on it.

:- pred node_a(links(numbers, int)::in, int::in, io::di, io::uo) is det.
node_a(Links, Round, !IO) :-
    ( if Round > rounds then
        true
    else
        report_sending(array_size, Round, !IO),
        gpu_a(Links, Round, Sum, !IO),
        report_sum(Sum, array_size * Round, !IO),
        node_a(Links, Round + 1, !IO)
    ).

:- pred node_b(links(numbers, int)::in, int::in, io::di, io::uo) is det.
node_b(Links, Round, !IO) :-
    ( if Round > rounds then
        true
    else
        report_waiting(Round, !IO),
        gpu_b(Links, Size, Sum, !IO),
        report_received(Size, Sum, !IO),
        node_b(Links, Round + 1, !IO)
    ).

%---------------------------------------------------------------------------%
% What runs on the GPUs.

    % GPU A: makes an array, sends it to GPU B, and waits for the sum.
:- pred gpu_a(links(numbers, int)::in, int::in, int::out, io::di, io::uo)
    is det.
gpu_a(links(ToB, ToA), Round, Sum, !IO) :-
    %$omp target
    (
        Numbers = array.init(array_size, Round),
        send(ToB, Numbers, !IO),
        receive(ToA, any, after(10000), Answer, !IO),
        (
            Answer = yes(Sum)
        ;
            Answer = no,
            Sum = -1
        )
    ).

    % GPU B: waits for an array, adds it up, and sends the sum to GPU A.
:- pred gpu_b(links(numbers, int)::in, int::out, int::out, io::di, io::uo)
    is det.
gpu_b(links(ToB, ToA), Size, Sum, !IO) :-
    %$omp target
    (
        receive(ToB, any, after(10000), Got, !IO),
        (
            Got = yes(Numbers),
            Size = array.size(Numbers),
            Sum = sum_from(Numbers, 0, 0),
            send(ToA, Sum, !IO)
        ;
            Got = no,
            Size = 0,
            Sum = -1
        )
    ).

:- func sum_from(numbers, int, int) = int.
sum_from(Numbers, I, Sum) =
    ( if I >= array.size(Numbers) then
        Sum
    else
        sum_from(Numbers, I + 1, Sum + Numbers ^ elem(I))
    ).
