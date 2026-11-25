% DESIGN: Processes and mailboxes
% A receive over a selector whose mailboxes are all closed can never get
% a message: a run-time error at once, rather than a wait for the
% timeout.
% ORACLE: none
%
% RUN: merkc -O0 %s -o %t0
% RUN: not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:both closed
% OUTPUT-NEXT:Uncaught Merkurius exception:
% OUTPUT-NEXT:Software Error: receive: every mailbox is closed or another node's
% OUTPUT-NOT:{{.}}
%
:- module run_receive_all_closed.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module process, string.

main(!IO) :-
    new_mailbox(A, !IO),
    new_mailbox(B, !IO),
    close_mailbox(A, !IO),
    close_mailbox(B, !IO),
    io.write_string("both closed\n", !IO),
    Sel = selecting(A, (func(N) = N : int),
        selecting(B, (func(S) = string.length(S)), new_selector)),
    receive(Sel, after(1000), R, !IO),
    io.write_line(R, !IO).
