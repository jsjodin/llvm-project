% A sender keeps sending to a process that has ended, while the next process
% of the GPU takes its slot; that one must get none of the words.
% A local queue's state is one word (runtime/place.h, ProcessQueue): with its
% incarnation and openness in two, a sender to the ended process could see
% its incarnation with the next process's openness, between the two stores
% that opened the queue, and put its word there. Before the fix, about one
% round in ten on gfx90a had a word in the fresh process.
%
% Each round: a target takes one word, so that the stale sender has its
% route to the target's queue, and ends; the stale sender sends it K words;
% a fresh process, to which nobody sends, starts once the target's slot is
% free, and counts what it gets for a millisecond with none.
%
% The kinds run at the host here, and on the GPU in
% process-stale-sends-gpu.test.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t
% RUN: %libomp-env %t host 20 1000 | FileCheck %s --check-prefix=HOST --match-full-lines

:- module process_stale_sends.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

    % Takes one word, so that the sender has its route, and ends.
:- process target(mailbox(int)) receives int on gpu.

target(Report, Self, !IO) :-
    receive(Self, any, after(10000), _, !IO),
    send(Report, -1, !IO).

    % Sends To K words, and reports.
:- process stale(int, mailbox(int), mailbox(int)) receives int on gpu.

stale(K, To, Report, _, !IO) :-
    send_words(K, To, !IO),
    send(Report, -2, !IO).

:- pred send_words(int::in, mailbox(int)::in, io::di, io::uo) is det.

send_words(K, To, !IO) :-
    ( if K > 0 then
        send(To, K, !IO),
        send_words(K - 1, To, !IO)
    else
        true
    ).

    % Nobody sends to it: reports how many words it got anyway.
:- process fresh(mailbox(int)) receives int on gpu.

fresh(Report, Self, !IO) :-
    count(0, Report, Self, !IO).

:- pred count(int::in, mailbox(int)::in, mailbox(int)::in,
    io::di, io::uo) is det.

count(N, Report, Self, !IO) :-
    receive(Self, any, after(1), R, !IO),
    ( if R = message(_) then
        count(N + 1, Report, Self, !IO)
    else
        send(Report, N, !IO)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = [Where, RText, KText], string.to_int(RText, Rounds),
            string.to_int(KText, K) then
        Place = ( if Where = "gpu" then gpu(0) else host ),
        new_mailbox(Reports, !IO),
        new_mailbox(Nothing, !IO),
        rounds(Place, Rounds, K, Reports, Nothing, 0, Got, 0, Hit, !IO),
        io.format("rounds %d, stale words got %d, in %d rounds\n",
            [i(Rounds), i(Got), i(Hit)], !IO)
    else
        io.write_string("usage: host or gpu, rounds, sends\n", !IO)
    ).

:- pred rounds(place::in, int::in, int::in, mailbox(int)::in,
    mailbox(int)::in, int::in, int::out, int::in, int::out,
    io::di, io::uo) is det.

rounds(Place, R, K, Reports, Nothing, !Got, !Hit, !IO) :-
    ( if R > 0 then
        spawn(Place, target(Reports), T, !IO),
        spawn(Place, stale(K, T, Reports), _, !IO),
        % The target's report, and a moment for its slot to come free.
        receive(Reports, any, after(10000), _, !IO),
        receive(Nothing, any, after(1), _, !IO),
        spawn(Place, fresh(Reports), _, !IO),
        take(2, 0, N, Reports, !IO),
        !:Got = !.Got + N,
        !:Hit = !.Hit + ( if N > 0 then 1 else 0 ),
        rounds(Place, R - 1, K, Reports, Nothing, !Got, !Hit, !IO)
    else
        true
    ).

    % The stale sender's and the fresh process's reports, in either order:
    % the words the fresh one got.
:- pred take(int::in, int::in, int::out, mailbox(int)::in,
    io::di, io::uo) is det.

take(M, N0, N, Reports, !IO) :-
    ( if M > 0 then
        receive(Reports, any, after(10000), R, !IO),
        N1 = ( if R = message(X), X >= 0 then N0 + X else N0 ),
        take(M - 1, N1, N, Reports, !IO)
    else
        N = N0
    ).
% HOST:rounds 20, stale words got 0, in 0 rounds
