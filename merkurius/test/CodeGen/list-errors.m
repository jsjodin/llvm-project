% The list module's det_ predicates and functions, and take_upto with a
% negative count, throw the library's exceptions, with mmc's messages:
% det_index1 reports as det_index0, and the functions det_index0 and
% det_last as their predicates, as in mmc, and the functions
% det_index0_of_first_occurrence and det_index1_of_first_occurrence as
% themselves. Nothing catches them yet, so the program writes the exception
% to the standard error and exits with status 1. which/1 picks the case;
% det_split_list fails a count beyond the list's length and a negative one
% alike, and det_replace_nth a position past the end and one below 1 each
% with its own message.
% The output and messages are mmc's.
%
% RUN: merkc %s -o %t.0 && %t.0 | FileCheck %s --check-prefix=OUTPUT-0 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.0.O2 && %t.0.O2 | FileCheck %s --check-prefix=OUTPUT-0 --match-full-lines --strict-whitespace
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
% RUN: sed 's/^which(0)\./which(8)./' %s > %t.8.m && merkc %t.8.m -o %t.8
% RUN: not %t.8 2> %t.8.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-8 --match-full-lines --strict-whitespace < %t.8.err
% RUN: sed 's/^which(0)\./which(9)./' %s > %t.9.m && merkc %t.9.m -o %t.9
% RUN: not %t.9 2> %t.9.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-9 --match-full-lines --strict-whitespace < %t.9.err
% RUN: sed 's/^which(0)\./which(10)./' %s > %t.10.m && merkc %t.10.m -o %t.10
% RUN: not %t.10 2> %t.10.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-10 --match-full-lines --strict-whitespace < %t.10.err
% RUN: sed 's/^which(0)\./which(11)./' %s > %t.11.m && merkc %t.11.m -o %t.11
% RUN: not %t.11 2> %t.11.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-10 --match-full-lines --strict-whitespace < %t.11.err
% RUN: sed 's/^which(0)\./which(12)./' %s > %t.12.m && merkc %t.12.m -o %t.12
% RUN: not %t.12 2> %t.12.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-12 --match-full-lines --strict-whitespace < %t.12.err
% RUN: sed 's/^which(0)\./which(13)./' %s > %t.13.m && merkc %t.13.m -o %t.13
% RUN: not %t.13 2> %t.13.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-13 --match-full-lines --strict-whitespace < %t.13.err

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
        det_split_list(1, [1, 2, 3], S1, S2),
        io.write({S1, S2}, !IO),
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
    else if N = 7 then
        io.write(take_upto(-1, [1]) : list(int), !IO)
    else if N = 8 then
        io.write_int(det_index0_of_first_occurrence([1, 2], 3), !IO)
    else if N = 9 then
        io.write_int(det_index1_of_first_occurrence([1, 2], 3), !IO)
    else if N =< 11 then
        det_split_list(( if N = 10 then 4 else -1 ), [1, 2, 3], S1, S2),
        io.write({S1, S2}, !IO)
    else
        det_replace_nth([1, 2, 3], ( if N = 12 then 4 else 0 ), 9, R),
        io.write(R, !IO)
    ).

% OUTPUT-0:before
% OUTPUT-0-NEXT:1[2]23[]{[1], [2, 3]}
% OUTPUT-0-NEXT:after
% OUTPUT:before
% OUTPUT-NOT:{{.}}
% ERROR-1:Uncaught Merkurius exception:
% ERROR-1-NEXT:Software Error: function `list.det_head'/1: Unexpected: empty list
% ERROR-2:Uncaught Merkurius exception:
% ERROR-2-NEXT:Software Error: function `list.det_tail'/1: Unexpected: empty list
% ERROR-3:Uncaught Merkurius exception:
% ERROR-3-NEXT:Software Error: predicate `list.det_index0'/3: Unexpected: index out of range
% ERROR-4:Uncaught Merkurius exception:
% ERROR-4-NEXT:Software Error: predicate `list.det_index0'/3: Unexpected: index out of range
% ERROR-5:Uncaught Merkurius exception:
% ERROR-5-NEXT:Software Error: predicate `list.det_last'/2: Unexpected: empty list
% ERROR-6:Uncaught Merkurius exception:
% ERROR-6-NEXT:Software Error: predicate `list.det_last'/2: Unexpected: empty list
% ERROR-7:Uncaught Merkurius exception:
% ERROR-7-NEXT:Software Error: list.m: predicate `list.take_upto'/3: Unexpected: index is negative
% ERROR-8:Uncaught Merkurius exception:
% ERROR-8-NEXT:Software Error: function `list.det_index0_of_first_occurrence'/2: Unexpected: item not found
% ERROR-9:Uncaught Merkurius exception:
% ERROR-9-NEXT:Software Error: function `list.det_index1_of_first_occurrence'/2: Unexpected: item not found
% ERROR-10:Uncaught Merkurius exception:
% ERROR-10-NEXT:Software Error: predicate `list.det_split_list'/4: Unexpected: index out of range
% ERROR-12:Uncaught Merkurius exception:
% ERROR-12-NEXT:Software Error: predicate `list.det_replace_nth'/4: Unexpected: Cannot replace element whose index position is past the end of the list
% ERROR-13:Uncaught Merkurius exception:
% ERROR-13-NEXT:Software Error: predicate `list.det_replace_nth'/4: Unexpected: Cannot replace element whose index position is less than 1.
