% A target region holds up no worker (design document, "OpenMP cooperation",
% Target regions): the process runs it on a thread of the blocking pool, and
% goes back to the workers after it. With one worker, the region here waits
% for a message from a process that can run only if the region left the worker
% free. Regions run one after another, reusing the pool, and in a spawned
% process.
%
% RUN: merkc -fopenmp %libomp-flags %s -o %t
% RUN: %libomp-env MERK_WORKERS=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env MERK_WORKERS=2 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && env MERK_WORKERS=1 %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% The program ends when main does, with a region still waiting, at once.
% RUN: sed 's/^ends_early = no\./ends_early = yes./' %s > %t.e.m && merkc -fopenmp %libomp-flags %t.e.m -o %t.e
% RUN: %libomp-env timeout 10 %t.e | FileCheck %s --check-prefix=EARLY --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module process_target.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

:- pred region(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
region(In, Out, !IO) :-
    %$omp target
    (
        receive(In, any, after(20000), Got, !IO),
        (
            Got = message(N),
            send(Out, N + 1, !IO)
        ;
            Got = timeout,
            send(Out, -1, !IO)
        )
    ).

:- pred sender(int::in, mailbox(int)::in, mailbox(int)::in, io::di, io::uo)
    is det.
sender(N, In, _, !IO) :-
    send(In, N, !IO).

:- pred child(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
child(Out, Box, !IO) :-
    region(Box, Out, !IO).

:- pred rounds(int::in, mailbox(int)::in, mailbox(int)::in, io::di, io::uo)
    is det.
rounds(N, In, Out, !IO) :-
    ( if N = 0 then
        true
    else
        spawn(sender(N, In), _, !IO),
        region(In, Out, !IO),
        receive(Out, any, infinity, Got, !IO),
        io.write_line(Got, !IO),
        rounds(N - 1, In, Out, !IO)
    ).

:- type answer
    --->    no
    ;       yes.

:- func ends_early = answer.
ends_early = no.

main(!IO) :-
    new_mailbox(In, !IO),
    new_mailbox(Out, !IO),
    ( if ends_early = yes then
        spawn(child(Out), _, !IO),
        new_mailbox(Idle : mailbox(int), !IO),
        receive(Idle, any, after(200), _, !IO),
        io.write_string("main ended\n", !IO)
    else
        everything(In, Out, !IO)
    ).

:- pred everything(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
everything(In, Out, !IO) :-
    rounds(3, In, Out, !IO),
    spawn(child(Out), Box, !IO),
    send(Box, 10, !IO),
    receive(Out, any, infinity, Got, !IO),
    io.write_line(Got, !IO).

% EARLY:main ended
% EARLY-NOT:{{.}}

% OUTPUT:message(4)
% OUTPUT-NEXT:message(3)
% OUTPUT-NEXT:message(2)
% OUTPUT-NEXT:message(11)
% OUTPUT-NOT:{{.}}
