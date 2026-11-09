% Target regions in several processes at once (docs/INPROGRESS.md, The
% scheduler, step 8): each process runs a region on device D that answers main
% over its own mailboxes, while the others do, and regions end at different
% times. main takes the answers in turn, so all of them are running together.
%
% RUN: merkc -fopenmp %libomp-flags %s -o %t
% RUN: %libomp-env MERK_WORKERS=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module process_target_many.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process.

:- func device_number = int.
device_number = 0.

:- pred answer(int::in, mailbox(int)::in, mailbox(int)::in, io::di, io::uo)
    is det.
answer(N, Box, Out, !IO) :-
    ( if N = 0 then
        true
    else
        receive(Box, any, after(20000), Got, !IO),
        (
            Got = message(V),
            send(Out, V + 1, !IO),
            answer(N - 1, Box, Out, !IO)
        ;
            Got = timeout,
            send(Out, -1000000, !IO)
        )
    ).

:- pred child(int::in, mailbox(int)::in, mailbox(int)::in, io::di, io::uo)
    is det.
child(Rounds, Out, Box, !IO) :-
    D = device_number,
    %$omp target device(D)
    (
        answer(Rounds, Box, Out, !IO)
    ).

:- type peer
    --->    peer(int, mailbox(int), mailbox(int)).

:- pred start(int::in, int::in, list(peer)::in, list(peer)::out,
    io::di, io::uo) is det.
start(K, Rounds, !Peers, !IO) :-
    ( if K = 0 then
        true
    else
        new_mailbox(Out, !IO),
        % A later process's region ends later.
        spawn(child(Rounds + K, Out), Box, !IO),
        !:Peers = [peer(Rounds + K, Box, Out) | !.Peers],
        start(K - 1, Rounds, !Peers, !IO)
    ).

:- pred ask_all(list(peer)::in, int::in, io::di, io::uo) is det.
ask_all([], _, !IO).
ask_all([peer(Limit, Box, _) | Peers], Round, !IO) :-
    ( if Round < Limit then
        send(Box, Round, !IO)
    else
        true
    ),
    ask_all(Peers, Round, !IO).

:- pred collect(list(peer)::in, int::in, int::in, int::out, io::di, io::uo)
    is det.
collect([], _, !Sum, !IO).
collect([peer(Limit, _, Out) | Peers], Round, !Sum, !IO) :-
    ( if Round < Limit then
        receive(Out, any, after(30000), Got, !IO),
        (
            Got = message(V),
            !:Sum = !.Sum + V
        ;
            Got = timeout,
            !:Sum = !.Sum - 1000000
        )
    else
        true
    ),
    collect(Peers, Round, !Sum, !IO).

:- pred rounds(int::in, int::in, list(peer)::in, int::in, int::out,
    io::di, io::uo) is det.
rounds(Round, End, Peers, !Sum, !IO) :-
    ( if Round >= End then
        true
    else
        ask_all(Peers, Round, !IO),
        collect(Peers, Round, !Sum, !IO),
        rounds(Round + 1, End, Peers, !Sum, !IO)
    ).

main(!IO) :-
    start(4, 100, [], Peers, !IO),
    rounds(0, 110, Peers, 0, Sum, !IO),
    io.write_int(Sum, !IO),
    io.nl(!IO).

% Rounds 0 to 100+K-1 of each of K = 1..4: the sum of (Round + 1).
% OUTPUT:21220
% OUTPUT-NOT:{{.}}
