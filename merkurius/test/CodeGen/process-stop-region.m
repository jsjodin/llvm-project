% process.stop of a process in a parallel region (docs/INPROGRESS.md,
% Failure, step 1): libomp's team cannot be abandoned, so the region ends
% first and the process ends at its next check, before it sends its sum.
% Its monitor says killed, and the team is whole: the main process's own
% region afterwards gives the right sum. Without -fopenmp the loop is
% sequential and the process ends inside it; the output is the same. mmc
% has no processes, so the output is written by hand.
%
% REQUIRES: libomp
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t.O0 && %libomp-env OMP_NUM_THREADS=4 %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_CHECK_MESSAGES=1 %libomp-env OMP_NUM_THREADS=4 %t 2>&1 >/dev/null | FileCheck %s --check-prefix=MESSAGES --match-full-lines
% MESSAGES:Merkurius runtime: 0 messages are neither read nor freed

:- module process_stop_region.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- func sum(int) = int.
sum(N) = S :-
    some [!S] (
        !:S = 0,
        %$omp parallel do reduction((+) : S)
        ( for(I, 1, N) do !:S = !.S + I mod 7 ),
        S = !.S
    ).

:- type happening
    --->    started
    ;       total(int)
    ;       went(down_info).

:- pred summer(mailbox(happening)::in, mailbox(int)::in, io::di, io::uo) is det.
summer(Events, Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = message(N),
        send(Events, started, !IO),
        S = sum(N),
        send(Events, total(S), !IO)
    ;
        Got = timeout
    ).

:- pred show(mailbox(happening)::in, io::di, io::uo) is det.
show(Events, !IO) :-
    receive(Events, any, after(60000), Got, !IO),
    (
        Got = message(started),
        io.write_string("started\n", !IO)
    ;
        Got = message(total(S)),
        io.format("total %d\n", [i(S)], !IO)
    ;
        Got = message(went(Down)),
        io.write_string("summer: " ++ reason_text(reason(Down)) ++ "\n", !IO)
    ;
        Got = timeout,
        io.write_string("nothing\n", !IO)
    ).

:- func went_as(down_info) = happening.
went_as(Down) = went(Down).

% OUTPUT:started
% OUTPUT-NEXT:summer: killed
% OUTPUT-NEXT:main's sum 2999998
% OUTPUT-NOT:{{.}}
main(!IO) :-
    new_mailbox(Events, !IO),
    spawn(summer(Events), Summer, !IO),
    monitor(Summer, Events, went_as, _, !IO),
    send(Summer, 2000000000, !IO),
    show(Events, !IO),
    stop(Summer, !IO),
    show(Events, !IO),
    io.format("main's sum %d\n", [i(sum(1000000))], !IO),
    close_mailbox(Events, !IO).
