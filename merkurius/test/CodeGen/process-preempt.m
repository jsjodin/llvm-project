% A process that never blocks, on one worker, lets the others run: a tail
% recursion that does not allocate, one that does, and a loop LLVM made of a
% recursion. Timeouts fire while it runs. The program ends when main does.
% (design document, "Processes and scheduler", Preemption)
%
% RUN: merkc -O0 %s -o %t.0 && env MERK_WORKERS=1 %t.0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.2 && env MERK_WORKERS=1 %t.2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_WORKERS=2 %t.2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_preempt.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process.

:- pred spin(int::in, mailbox(int)::in, io::di, io::uo) is det.
spin(N, Box, !IO) :-
    ( if N < 0 then
        send(Box, N, !IO)
    else
        spin((N + 1) mod 1000, Box, !IO)
    ).

:- pred churn(list(int)::in, mailbox(int)::in, io::di, io::uo) is det.
churn(L, Box, !IO) :-
    ( if L = [-1 | _] then
        send(Box, 0, !IO)
    else
        churn([1, 2, 3], Box, !IO)
    ).

:- func sum_down(int, int) = int.
sum_down(N, Mod) =
    (if N = 0 then 0 else (N + sum_down(N - 1, Mod)) mod Mod).

:- pred summer(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
summer(Reply, _, !IO) :-
    send(Reply, sum_down(500000, 1000003), !IO).

:- pred ticks(int::in, mailbox(int)::in, io::di, io::uo) is det.
ticks(N, Box, !IO) :-
    ( if N = 0 then
        true
    else
        receive(Box, any, after(2), _, !IO),
        ticks(N - 1, Box, !IO)
    ).

main(!IO) :-
    new_mailbox(Reply, !IO),
    spawn(spin(0), _ : mailbox(int), !IO),
    spawn(churn([]), _ : mailbox(int), !IO),
    ticks(10, Reply, !IO),
    io.write_string("timeouts fired\n", !IO),
    spawn(summer(Reply), _ : mailbox(int), !IO),
    receive(Reply, any, infinity, Sum, !IO),
    io.write_line(Sum, !IO).

% OUTPUT:timeouts fired
% OUTPUT-NEXT:message({{[0-9]+}})
% OUTPUT-NOT:{{.}}
