% The list module's det_ predicates and functions, and take_upto with a
% negative count, throw the library's exceptions, with mmc's messages:
% det_index1 reports as det_index0, and the functions det_index0 and
% det_last as their predicates, as in mmc. Nothing catches them yet, so
% the program writes the exception to the standard error and exits with
% status 1. which/1 picks the case. The output and messages are mmc's.
%
% RUN: merkc %s -o %t.0 && %t.0 | FileCheck %s --check-prefix=OUTPUT-0 --match-full-lines --strict-whitespace
% RUN: sed 's/^which(0)\./which(1)./' %s > %t.1.m && merkc %t.1.m -o %t.1
% RUN: not %t.1 2> %t.1.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-1 --match-full-lines --strict-whitespace < %t.1.err
% RUN: sed 's/^which(0)\./which(2)./' %s > %t.2.m && merkc %t.2.m -o %t.2
% RUN: not %t.2 2> %t.2.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-2 --match-full-lines --strict-whitespace < %t.2.err
% RUN: sed 's/^which(0)\./which(3)./' %s > %t.3.m && merkc %t.3.m -o %t.3
% RUN: not %t.3 2> %t.3.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-3 --match-full-lines --strict-whitespace < %t.3.err
% RUN: sed 's/^which(0)\./which(4)./' %s > %t.4.m && merkc %t.4.m -o %t.4
% RUN: not %t.4 2> %t.4.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-4 --match-full-lines --strict-whitespace < %t.4.err
% RUN: sed 's/^which(0)\./which(5)./' %s > %t.5.m && merkc %t.5.m -o %t.5
% RUN: not %t.5 2> %t.5.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-5 --match-full-lines --strict-whitespace < %t.5.err
% RUN: sed 's/^which(0)\./which(6)./' %s > %t.6.m && merkc %t.6.m -o %t.6
% RUN: not %t.6 2> %t.6.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-6 --match-full-lines --strict-whitespace < %t.6.err
% RUN: sed 's/^which(0)\./which(7)./' %s > %t.7.m && merkc %t.7.m -o %t.7
% RUN: not %t.7 2> %t.7.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-7 --match-full-lines --strict-whitespace < %t.7.err

:- module list_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred which(int::out) is det.
which(0).

main(!IO) :-
    io.write_string("before\n", !IO),
    which(N),
    run(N, [], !IO).

:- pred run(int::in, list(int)::in, io::di, io::uo) is det.
run(N, E, !IO) :-
    ( if N = 0 then
        io.write_int(det_head([1]), !IO),
        io.write(det_tail([1, 2]), !IO),
        io.write_int(det_index1([1, 2], 2), !IO),
        io.write_int(det_last([1, 2, 3]), !IO),
        io.write(take_upto(0, [1]) : list(int), !IO),
        io.write_string("\nafter\n", !IO)
    else if N = 1 then
        io.write_int(det_head(E), !IO)
    else if N = 2 then
        io.write(det_tail(E), !IO)
    else if N = 3 then
        io.write_int(det_index0([1, 2], 2), !IO)
    else if N = 4 then
        det_index1([1, 2], 0, X),
        io.write_int(X, !IO)
    else if N = 5 then
        io.write_int(det_last(E), !IO)
    else if N = 6 then
        det_last(E, Y),
        io.write_int(Y, !IO)
    else
        io.write(take_upto(-1, [1]) : list(int), !IO)
    ).

% OUTPUT-0:before
% OUTPUT-0-NEXT:1[2]23[]
% OUTPUT-0-NEXT:after
% OUTPUT:before
% OUTPUT-NOT:{{.}}
% ERROR-1:Uncaught Mercury exception:
% ERROR-1-NEXT:Software Error: function `list.det_head'/1: Unexpected: empty list
% ERROR-2:Uncaught Mercury exception:
% ERROR-2-NEXT:Software Error: function `list.det_tail'/1: Unexpected: empty list
% ERROR-3:Uncaught Mercury exception:
% ERROR-3-NEXT:Software Error: predicate `list.det_index0'/3: Unexpected: index out of range
% ERROR-4:Uncaught Mercury exception:
% ERROR-4-NEXT:Software Error: predicate `list.det_index0'/3: Unexpected: index out of range
% ERROR-5:Uncaught Mercury exception:
% ERROR-5-NEXT:Software Error: predicate `list.det_last'/2: Unexpected: empty list
% ERROR-6:Uncaught Mercury exception:
% ERROR-6-NEXT:Software Error: predicate `list.det_last'/2: Unexpected: empty list
% ERROR-7:Uncaught Mercury exception:
% ERROR-7-NEXT:Software Error: list.m: predicate `list.take_upto'/3: Unexpected: index is negative
