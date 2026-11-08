% A failure ends only its process (design document, "Exceptions, links and
% monitors"): an uncaught exception and a stack overflow in spawned processes
% are reported, and the rest of the program goes on, and exits with status 0.
% The same exception in main ends the program with status 1.
%
% RUN: merkc -O0 %s -o %t.0 && %t.0 2> %t.0.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=REPORTS --match-full-lines --strict-whitespace < %t.0.err
% RUN: merkc -O2 %s -o %t.2 && %t.2 2> %t.2.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=REPORTS --match-full-lines --strict-whitespace < %t.2.err
% RUN: env MERK_WORKERS=1 %t.2 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: sed 's/^in_main = no\./in_main = yes./' %s > %t.m.m && merkc %t.m.m -o %t.m
% RUN: not %t.m 2> %t.m.err | FileCheck %s --check-prefix=MAIN --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=REPORT1 --match-full-lines --strict-whitespace < %t.m.err

:- module process_failure.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, int, maybe, process.

:- func in_main = bool.
in_main = no.

:- pred divider(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
divider(Reply, Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = yes(N),
        send(Reply, 100 // N, !IO)
    ;
        Got = no
    ).

:- func deep(int) = int.
% Not `1 + deep(N + 1)`, which LLVM makes a loop that never overflows.
deep(N) = (deep(N + 1) >> 1) - N.

:- pred overflower(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
overflower(Reply, Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = yes(N),
        send(Reply, deep(N), !IO)
    ;
        Got = no
    ).

:- pred echo(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
echo(Reply, Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = yes(N),
        send(Reply, N + 1, !IO)
    ;
        Got = no
    ).

main(!IO) :-
    new_mailbox(Reply, !IO),
    ( if in_main = yes then
        io.write_string("before\n", !IO),
        io.write_int(100 // zero, !IO),
        io.nl(!IO)
    else
        new_mailbox(Downs, !IO),
        spawn(divider(Reply), Divider, !IO),
        monitor(Divider, Downs, (func(D) = D), _, !IO),
        send(Divider, 0, !IO),
        report(Downs, !IO),
        spawn_with_stack(1048576, overflower(Reply), Overflower, !IO),
        monitor(Overflower, Downs, (func(D) = D), _, !IO),
        send(Overflower, 0, !IO),
        report(Downs, !IO),
        % A message to a process that has failed goes nowhere.
        send(Divider, 1, !IO),
        spawn(echo(Reply), Echo, !IO),
        send(Echo, 41, !IO),
        receive(Reply, any, infinity, Late, !IO),
        io.write_line(Late, !IO)
    ).

:- pred report(mailbox(down_info)::in, io::di, io::uo) is det.
report(Downs, !IO) :-
    receive(Downs, any, infinity, Got, !IO),
    (
        Got = yes(Down),
        io.write_string(reason_text(reason(Down)), !IO),
        io.nl(!IO)
    ;
        Got = no,
        io.write_string("no down\n", !IO)
    ).

:- func zero = int.
:- pragma no_inline(func(zero/0)).
zero = 0.

% OUTPUT:domain_error("int.\'//\': division by zero")
% OUTPUT-NEXT:stack overflow
% OUTPUT-NEXT:yes(42)
% OUTPUT-NOT:{{.}}

% REPORTS-DAG:Uncaught Mercury exception:
% REPORTS-DAG:domain_error("int.\'//\': division by zero")
% REPORTS-DAG:Mercury runtime: stack overflow

% MAIN:before
% MAIN-NOT:{{.}}

% REPORT1:Uncaught Mercury exception:
% REPORT1-NEXT:domain_error("int.\'//\': division by zero")
