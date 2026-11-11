% A region's slot is used again by many regions, one after another: the
% host gives the slot a new generation each time, and the region's device
% clears its words as it starts if they are an earlier region's (device-
% ring.h). Each round, a target teams region whose teams pass numbers
% through team mailboxes, which put the region's table of teams in the
% slot, and then a target region that attaches a mailbox's ring, which goes
% in the slot's table of rings, and makes an array, in a chunk that an
% earlier region's chunk may be again, as the host keeps them. Each answer
% must be the round's: an earlier region's table, left in the slot, would
% point into a chunk that is now someone else's.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module omp_target_region_slot_reuse.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

main(!IO) :-
    new_mailbox(In : mailbox(int), !IO),
    new_mailbox(Out : mailbox(int), !IO),
    rounds(1, 300, In, Out, 0, Right, !IO),
    io.write_int(Right, !IO),
    io.write_string(" of 300 rounds were right\n", !IO).

:- pred rounds(int::in, int::in, mailbox(int)::in, mailbox(int)::in,
    int::in, int::out, io::di, io::uo) is det.
rounds(R, Last, In, Out, !Right, !IO) :-
    ( if R > Last then
        true
    else
        %$omp target teams num_teams(3)
        pass(R, Out, !IO),
        send(In, R, !IO),
        %$omp target
        sum_of(In, Out, !IO),
        receive(Out, any, after(10000), Teams, !IO),
        receive(Out, any, after(10000), Sum, !IO),
        ( if
            Teams = message(1),
            Sum = message(R * (R - 1) / 2)
        then
            !:Right = !.Right + 1
        else
            io.write_string("round " ++ int_to_string(R) ++ " was wrong\n", !IO)
        ),
        rounds(R + 1, Last, In, Out, !Right, !IO)
    ).

    % The teams other than team 0 send R * 10 + their number to team 0,
    % which sends 1 to Out if the sum is right, as it is from the teams of
    % this round, and 0 if not. Without OpenMP, team 0 is the only team.
    %
:- pred pass(int::in, mailbox(int)::in, io::di, io::uo) is det.
pass(R, Out, !IO) :-
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    team_mailbox(0, Zero, !IO),
    ( if Me = 0 then
        gather(Zero, All - 1, 0, Sum, !IO),
        ( if Sum = R * 10 * (All - 1) + (All - 1) * All / 2 then
            send(Out, 1, !IO)
        else
            send(Out, 0, !IO)
        )
    else
        send(Zero, R * 10 + Me, !IO)
    ).

:- pred gather(mailbox(int)::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
gather(Zero, N, !Sum, !IO) :-
    ( if N =< 0 then
        true
    else
        receive(Zero, any, after(10000), Got, !IO),
        (
            Got = message(X),
            !:Sum = !.Sum + X,
            gather(Zero, N - 1, !Sum, !IO)
        ;
            Got = timeout,
            !:Sum = -1
        )
    ).

    % The sum of 0 to N - 1, from an array of them, for N from In.
    %
:- pred sum_of(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
sum_of(In, Out, !IO) :-
    receive(In, any, after(10000), Got, !IO),
    (
        Got = message(N),
        array.init(N, 0, A0),
        ( for(I, 0, N - 1), fromto(A0, A1, A2, A) do
            A2 = A1 ^ elem(I) := I
        ),
        ( for(I, 0, N - 1), fromto(0, S0, S1, S) do
            S1 = S0 + A ^ elem(I)
        ),
        send(Out, S, !IO)
    ;
        Got = timeout,
        send(Out, -1, !IO)
    ).

% OUTPUT:300 of 300 rounds were right
