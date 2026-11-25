% DESIGN: Processes and mailboxes
% Only a mailbox's owner receives from it: another process may hold the
% mailbox and send to it, and a receive from it there stops the program.
% ORACLE: none
%
% RUN: merkc -O0 %s -o %t0
% RUN: not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:got the child's mailbox
% OUTPUT-NEXT:Merkurius runtime: a process received from a mailbox of another process
% OUTPUT-NOT:{{.}}
%
:- module run_receive_not_owner.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module process.

:- pred child(mailbox(mailbox(int))::in, mailbox(int)::in, io::di, io::uo)
    is det.

child(Back, Self, !IO) :-
    send(Back, Self, !IO),
    receive(Self, any, infinity, _, !IO).

main(!IO) :-
    new_mailbox(Back, !IO),
    spawn(child(Back), _, !IO),
    receive(Back, any, infinity, R, !IO),
    (
        R = message(Box),
        io.write_string("got the child's mailbox\n", !IO),
        receive(Box, any, after(1000), R2, !IO),
        io.write_line(R2, !IO)
    ;
        R = timeout
    ).
