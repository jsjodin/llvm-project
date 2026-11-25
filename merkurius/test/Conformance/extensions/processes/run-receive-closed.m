% DESIGN: Processes and mailboxes
% A send to a closed mailbox is dropped, and a receive from it can never
% get a message, which is a mistake in the program: a run-time error at
% once, rather than a wait for the timeout.
% ORACLE: none
%
% RUN: merkc -O0 %s -o %t0
% RUN: not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:closed
% OUTPUT-NEXT:Uncaught Merkurius exception:
% OUTPUT-NEXT:Software Error: receive: the mailbox is closed or another node's
% OUTPUT-NOT:{{.}}
%
:- module run_receive_closed.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module process.

main(!IO) :-
    new_mailbox(Box, !IO),
    send(Box, 1, !IO),
    close_mailbox(Box, !IO),
    % A send to a closed mailbox is dropped.
    send(Box, 2, !IO),
    io.write_string("closed\n", !IO),
    receive(Box, any, after(1000), R, !IO),
    io.write_line(R : received(int), !IO).
