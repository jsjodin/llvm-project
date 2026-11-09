% More messages than a mailbox's ring holds (64), and the mailbox passing
% between regions and the host: the first region takes 3, main takes the
% next, and the second region takes the rest, all in order of sending
% (docs/INPROGRESS.md, The device's side of the first slice, step 7). Each
% message counts by its place, so one out of order changes the sum. Then a
% region takes turns between two mailboxes whose slots fall in the same place
% of the device's table of rings, 64 apart.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_receive_ring.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

main(!IO) :-
    new_mailbox(In : mailbox(int), !IO),
    new_mailbox(Out : mailbox(int), !IO),
    send_from(1, 200, In, !IO),
    %$omp target
    (
        take(In, 1, 3, 0, First, !IO),
        send(Out, First, !IO)
    ),
    receive(In, any, after(0), Next, !IO),
    %$omp target
    (
        rest(In, 5, 0, Rest, !IO),
        send(Out, Rest, !IO)
    ),
    receive(In, any, after(0), Late, !IO),
    receive(Out, any, after(0), R1, !IO),
    receive(Out, any, after(0), R2, !IO),
    io.print_line({R1, Next, R2, Late}, !IO),
    new_mailbox(A : mailbox(int), !IO),
    more_mailboxes(63, !IO),
    new_mailbox(B : mailbox(int), !IO),
    send(A, 1, !IO),
    send(B, 2, !IO),
    send(A, 3, !IO),
    send(B, 4, !IO),
    %$omp target
    (
        receive(A, any, after(0), X1, !IO),
        receive(B, any, after(0), X2, !IO),
        receive(A, any, after(0), X3, !IO),
        receive(B, any, after(0), X4, !IO),
        send(Out, 1000 * got(X1) + 100 * got(X2) + 10 * got(X3) + got(X4), !IO)
    ),
    receive(Out, any, after(0), R3, !IO),
    io.print_line(R3, !IO).

:- func got(received(int)) = int.
got(timeout) = 0.
got(message(N)) = N.

:- pred more_mailboxes(int::in, io::di, io::uo) is det.
more_mailboxes(N, !IO) :-
    ( if N = 0 then
        true
    else
        new_mailbox(_ : mailbox(int), !IO),
        more_mailboxes(N - 1, !IO)
    ).

% The sum of the next messages, which must be Place to Last; -1 otherwise.
:- pred take(mailbox(int)::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
take(In, Place, Last, !Sum, !IO) :-
    ( if Place > Last then
        true
    else
        receive(In, any, after(0), Got, !IO),
        ( if Got = message(Place) then
            !:Sum = !.Sum + Place,
            take(In, Place + 1, Last, !Sum, !IO)
        else
            !:Sum = -1
        )
    ).

% The sum of all that is left, which must count on from Place; -1 otherwise.
:- pred rest(mailbox(int)::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
rest(In, Place, !Sum, !IO) :-
    receive(In, any, after(0), Got, !IO),
    (
        Got = timeout
    ;
        Got = message(N),
        ( if N = Place then
            !:Sum = !.Sum + N,
            rest(In, Place + 1, !Sum, !IO)
        else
            !:Sum = -1
        )
    ).

:- pred send_from(int::in, int::in, mailbox(int)::in, io::di, io::uo) is det.
send_from(I, N, M, !IO) :-
    ( if I > N then
        true
    else
        send(M, I, !IO),
        send_from(I + 1, N, M, !IO)
    ).

% OUTPUT:{message(6), message(4), message(20090), timeout}
% OUTPUT-NEXT:message(1234)
