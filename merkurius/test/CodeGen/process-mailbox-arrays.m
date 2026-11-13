% An array of mailboxes in a GPU process's code (design document, Messages
% and memory on the device): each of a ring of nodes keeps every node's
% mailbox, which the host sends it one at a time, as peer(J, Mailbox), in an
% array it makes, and passes a token on to the next node by it, round the
% ring three times; each node then tells the host the sum of the tokens it
% saw. Such an array is a value the process makes, not a message: it holds
% the handles it was given, which stay in the messages that brought them.
% The nodes run at the host here, and on the GPU in
% process-mailbox-arrays-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand: node J sees the tokens J, J + N and J + 2N, or N, 2N and 3N for
% node 0.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

% CHECK:node 0: 30
% CHECK-NEXT:node 1: 18
% CHECK-NEXT:node 2: 21
% CHECK-NEXT:node 3: 24
% CHECK-NEXT:node 4: 27

:- module process_mailbox_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- type to_node
    --->    peer(int, mailbox(to_node))
    ;       token(int).

:- type to_host
    --->    seen(int, int).

    % Node J of N: its peers, then Rounds tokens.
:- process node(mailbox(to_host), int, int, int) receives to_node on gpu.

node(Host, J, N, Rounds, Self, !IO) :-
    array.init(N, Self, Peers0),
    peers(Self, N, Peers0, Peers, !IO),
    ( if J = 0 then
        send(Peers ^ elem(1 mod N), token(1), !IO)
    else
        true
    ),
    tokens(Self, Peers, J, N, N * Rounds, Rounds, 0, Sum, !IO),
    send(Host, seen(J, Sum), !IO).

    % The N nodes' mailboxes, by their numbers.
:- pred peers(mailbox(to_node)::in, int::in,
    array(mailbox(to_node))::array_di, array(mailbox(to_node))::array_uo,
    io::di, io::uo) is det.

peers(Self, Left, !Peers, !IO) :-
    ( if Left > 0 then
        receive(Self, any, infinity, R, !IO),
        ( if R = message(peer(K, M)) then
            !Peers ^ elem(K) := M
        else
            true
        ),
        peers(Self, Left - 1, !Peers, !IO)
    else
        true
    ).

    % Takes Left tokens, each passed on to the next node but the last of
    % all, Last.
:- pred tokens(mailbox(to_node)::in, array(mailbox(to_node))::in, int::in,
    int::in, int::in, int::in, int::in, int::out, io::di, io::uo) is det.

tokens(Self, Peers, J, N, Last, Left, !Sum, !IO) :-
    ( if Left > 0 then
        receive(Self, any, infinity, R, !IO),
        ( if R = message(token(K)) then
            !:Sum = !.Sum + K,
            ( if K < Last then
                send(Peers ^ elem((J + 1) mod N), token(K + 1), !IO)
            else
                true
            )
        else
            true
        ),
        tokens(Self, Peers, J, N, Last, Left - 1, !Sum, !IO)
    else
        true
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    N = 5,
    new_mailbox(Host, !IO),
    list.map_foldl(spawn_node(Place, Host, N), 0 .. N - 1, Nodes, !IO),
    list.foldl(introduce(Nodes), Nodes, !IO),
    list.map_foldl(seen(Host), Nodes, Sums, !IO),
    list.foldl(write_sum, list.sort(Sums), !IO).

:- pred spawn_node(place::in, mailbox(to_host)::in, int::in, int::in,
    mailbox(to_node)::out, io::di, io::uo) is det.

spawn_node(Place, Host, N, J, Node, !IO) :-
    spawn(Place, node(Host, J, N, 3), Node, !IO).

    % Sends Node every node's mailbox, the last node's first.
:- pred introduce(list(mailbox(to_node))::in, mailbox(to_node)::in,
    io::di, io::uo) is det.

introduce(Nodes, Node, !IO) :-
    list.foldl(send_peer(Node), list.reverse(number(0, Nodes)), !IO).

:- func number(int, list(T)) = list({int, T}).

number(_, []) = [].
number(K, [X | Xs]) = [{K, X} | number(K + 1, Xs)].

:- pred send_peer(mailbox(to_node)::in, {int, mailbox(to_node)}::in,
    io::di, io::uo) is det.

send_peer(Node, {K, M}, !IO) :-
    send(Node, peer(K, M), !IO).

:- pred seen(mailbox(to_host)::in, mailbox(to_node)::in, {int, int}::out,
    io::di, io::uo) is det.

seen(Host, _, Seen, !IO) :-
    receive(Host, any, after(60000), R, !IO),
    ( if R = message(seen(J, Sum)) then
        Seen = {J, Sum}
    else
        Seen = {-1, -1}
    ).

:- pred write_sum({int, int}::in, io::di, io::uo) is det.

write_sum({J, Sum}, !IO) :-
    io.format("node %d: %d\n", [i(J), i(Sum)], !IO).
