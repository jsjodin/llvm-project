% Several process kinds that run loops across teams at once on one GPU, each
% with a share of the kernel's teams (process.set_worker_teams;
% docs/INPROGRESS.md, step 4 of the demo, (m1)). With share(K), each worker
% spawned takes up to a K'th of the kernel's teams, so K of them can run
% loops at once, where the first would otherwise take every free team
% (process-loops-gpu.test). Each worker is spawned before any is asked, so
% they all hold their teams together, and each sums the squares of a range
% of its own, in a loop with a reduction, three times, every worker asked
% before any answer is read, so that their loops overlap; the fourth's sum
% wraps, as int arithmetic does. At the host the setting does nothing. The program runs its kinds
% at the host here, and at the GPU in process-worker-teams-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host 2 | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=TWO
% RUN: %libomp-env %t2 host 4 | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=FOUR
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host 2 | FileCheck %s --match-full-lines --strict-whitespace --check-prefix=TWO
% RUN: not %libomp-env %t2 host 0 2>&1 | FileCheck %s --check-prefix=ZERO

% TWO:worker 0: 333333833333500000 333333833333500000 333333833333500000
% TWO-NEXT:worker 1: 2666668666667000000 2666668666667000000 2666668666667000000
% TWO-NEXT:done
% FOUR:worker 0: 333333833333500000 333333833333500000 333333833333500000
% FOUR-NEXT:worker 1: 2666668666667000000 2666668666667000000 2666668666667000000
% FOUR-NEXT:worker 2: 9000004500000500000 9000004500000500000 9000004500000500000
% FOUR-NEXT:worker 3: 2886597259624448384 2886597259624448384 2886597259624448384
% FOUR-NEXT:done
% ZERO:Software Error: process.set_worker_teams: share(0): a share is of at least 1

:- module process_worker_teams.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

    % Answers each N with the sum of the squares of the integers from 1 to
    % N, until 0.
:- process worker(mailbox(int)) receives int on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N > 0 then
        %$omp distribute parallel do reduction((+) : Sum)
        ( for(I, 1, N), fromto(0, S0, S1, Sum) do
            S1 = S0 + I * I
        ),
        send(Reply, Sum, !IO),
        worker(Reply, Self, !IO)
    else
        true
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if
        Args = [Where, KText],
        string.to_int(KText, K),
        ( Where = "host", Place = host
        ; Where = "gpu", Place = gpu(0)
        )
    then
        set_worker_teams(Place, share(K), !IO),
        Workers = 0 .. K - 1,
        list.map_foldl(start(Place), Workers, Started, !IO),
        list.foldl(ask, Started, !IO),
        list.foldl(answers, Started, !IO),
        list.foldl(stop, Started, !IO),
        io.write_string("done\n", !IO)
    else if Args = ["gpu-third"] then
        % With a share of 2, the first two workers leave no team free.
        set_worker_teams(gpu(0), share(2), !IO),
        list.map_foldl(start(gpu(0)), [0, 1, 2], _, !IO)
    else
        io.write_string("usage: host K | gpu K | gpu-third\n", !IO)
    ).

    % A worker's number, the mailbox it answers to, and its own.
:- type worker
    --->    worker(int, mailbox(int), mailbox(int)).

:- pred start(place::in, int::in, worker::out, io::di, io::uo) is det.

start(Place, J, worker(J, Reply, W), !IO) :-
    new_mailbox(Reply, !IO),
    spawn(Place, worker(Reply), W, !IO).

    % Worker J sums the squares up to (J + 1) * 10^6, three times; every
    % worker is asked before any answer is read.
:- pred ask(worker::in, io::di, io::uo) is det.

ask(worker(J, _, W), !IO) :-
    N = (J + 1) * 1000000,
    send(W, N, !IO),
    send(W, N, !IO),
    send(W, N, !IO).

:- pred answers(worker::in, io::di, io::uo) is det.

answers(worker(J, Reply, _), !IO) :-
    receive(Reply, any, after(60000), A, !IO),
    receive(Reply, any, after(60000), B, !IO),
    receive(Reply, any, after(60000), C, !IO),
    io.format("worker %d: %s %s %s\n", [i(J), s(text(A)), s(text(B)),
        s(text(C))], !IO).

:- func text(received(int)) = string.

text(message(N)) = string.int_to_string(N).
text(timeout) = "timeout".

:- pred stop(worker::in, io::di, io::uo) is det.

stop(worker(_, _, W), !IO) :-
    send(W, 0, !IO).
