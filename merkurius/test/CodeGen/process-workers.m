% Processes on several workers (design document, "Processes and scheduler"):
% each allocates before and after waiting, so that one that moves to another
% thread must go on with its own heap. The result does not depend on how
% many workers there are.
%
% RUN: merkc -O2 %s -o %t
% RUN: env MERK_WORKERS=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_WORKERS=2 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_WORKERS=8 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: not env MERK_WORKERS=0 %t 2>&1 | FileCheck %s --check-prefix=BAD --match-full-lines --strict-whitespace

:- module process_workers.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process.

:- type token
    --->    token(int, list(int)).

:- pred relay(int::in, mailbox(token)::in, mailbox(int)::in,
    mailbox(token)::in, io::di, io::uo) is det.
relay(Left, Next, Done, Box, !IO) :-
    Before = 1 .. 50,
    receive(Box, any, infinity, Got, !IO),
    ( if Got = yes(token(N, Seen)) then
        After = 1 .. 50,
        Added = list.foldl(int.plus, Before, 0)
            - list.foldl(int.plus, After, 0) + 1,
        send(Next, token(N + Added, [Added | Seen]), !IO),
        ( if Left > 1 then
            relay(Left - 1, Next, Done, Box, !IO)
        else
            send(Done, 1, !IO)
        )
    else
        true
    ).

:- pred ring(int::in, int::in, mailbox(token)::in, mailbox(int)::in,
    mailbox(token)::out, io::di, io::uo) is det.
ring(Count, Laps, Last, Done, First, !IO) :-
    ( if Count = 0 then
        First = Last
    else
        spawn(relay(Laps, Last, Done), Box, !IO),
        ring(Count - 1, Laps, Box, Done, First, !IO)
    ).

    % Tokens go round; each lap ends here and starts again.
:- pred laps(int::in, mailbox(token)::in, mailbox(token)::in,
    int::in, int::out, io::di, io::uo) is det.
laps(Left, First, Home, !Sum, !IO) :-
    ( if Left = 0 then
        true
    else
        send(First, token(0, []), !IO),
        receive(Home, any, infinity, Got, !IO),
        ( if Got = yes(token(N, Seen)), N = list.length(Seen) then
            !:Sum = !.Sum + N
        else
            !:Sum = -1000000000
        ),
        laps(Left - 1, First, Home, !Sum, !IO)
    ).

:- pred collect(int::in, mailbox(int)::in, int::in, int::out,
    io::di, io::uo) is det.
collect(Left, Done, !Sum, !IO) :-
    ( if Left = 0 then
        true
    else
        receive(Done, any, infinity, Got, !IO),
        ( if Got = yes(N) then !:Sum = !.Sum + N else true ),
        collect(Left - 1, Done, !Sum, !IO)
    ).

main(!IO) :-
    new_mailbox(Home, !IO),
    new_mailbox(Done, !IO),
    ring(100, 50, Home, Done, First, !IO),
    laps(50, First, Home, 0, Sum, !IO),
    io.write_int(Sum, !IO),
    io.nl(!IO),
    collect(100, Done, 0, Ended, !IO),
    io.write_int(Ended, !IO),
    io.nl(!IO).

% OUTPUT:5000
% OUTPUT-NEXT:100
% OUTPUT-NOT:{{.}}

% BAD:Mercury runtime: MERK_WORKERS=0 is not a positive number
