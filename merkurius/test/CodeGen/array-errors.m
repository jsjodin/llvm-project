% The array predicates' checks throw the library's exceptions, with mmc's
% messages, which nothing catches yet: the program writes the exception to
% the standard error and exits with status 1. which/1 picks the case, one
% for each check; its no_inline keeps the case from being a constant.
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
% RUN: FileCheck %s --check-prefix=ERROR-11 --match-full-lines --strict-whitespace < %t.11.err
% RUN: sed 's/^which(0)\./which(12)./' %s > %t.12.m && merkc %t.12.m -o %t.12
% RUN: not %t.12 2> %t.12.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-12 --match-full-lines --strict-whitespace < %t.12.err
% RUN: sed 's/^which(0)\./which(13)./' %s > %t.13.m && merkc %t.13.m -o %t.13
% RUN: not %t.13 2> %t.13.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-13 --match-full-lines --strict-whitespace < %t.13.err
% RUN: sed 's/^which(0)\./which(14)./' %s > %t.14.m && merkc %t.14.m -o %t.14
% RUN: not %t.14 2> %t.14.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-14 --match-full-lines --strict-whitespace < %t.14.err
% RUN: sed 's/^which(0)\./which(15)./' %s > %t.15.m && merkc %t.15.m -o %t.15
% RUN: not %t.15 2> %t.15.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-15 --match-full-lines --strict-whitespace < %t.15.err
% RUN: sed 's/^which(0)\./which(16)./' %s > %t.16.m && merkc %t.16.m -o %t.16
% RUN: not %t.16 2> %t.16.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-16 --match-full-lines --strict-whitespace < %t.16.err
% RUN: sed 's/^which(0)\./which(17)./' %s > %t.17.m && merkc %t.17.m -o %t.17
% RUN: not %t.17 2> %t.17.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-17 --match-full-lines --strict-whitespace < %t.17.err
% RUN: sed 's/^which(0)\./which(18)./' %s > %t.18.m && merkc %t.18.m -o %t.18
% RUN: not %t.18 2> %t.18.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-18 --match-full-lines --strict-whitespace < %t.18.err

% OUTPUT-0:before
% OUTPUT-0-NEXT:4after
% OUTPUT-0-NOT:{{.}}

% OUTPUT:before
% OUTPUT-NOT:{{.}}

% ERROR-1:Uncaught Mercury exception:
% ERROR-1-NEXT:index_out_of_bounds("array.lookup: index 4 not in range [0, 3]")
% ERROR-1-NOT:{{.}}
% ERROR-2:Uncaught Mercury exception:
% ERROR-2-NEXT:index_out_of_bounds("array.lookup: index -1 not in range [0, 3]")
% ERROR-2-NOT:{{.}}
% ERROR-3:Uncaught Mercury exception:
% ERROR-3-NEXT:index_out_of_bounds("array.set: index 4 not in range [0, 3]")
% ERROR-3-NOT:{{.}}
% ERROR-4:Uncaught Mercury exception:
% ERROR-4-NEXT:index_out_of_bounds("array.set: index 9 not in range [0, 3]")
% ERROR-4-NOT:{{.}}
% ERROR-5:Uncaught Mercury exception:
% ERROR-5-NEXT:index_out_of_bounds("first argument of array.swap: index 5 not in range [0, 3]")
% ERROR-5-NOT:{{.}}
% ERROR-6:Uncaught Mercury exception:
% ERROR-6-NEXT:index_out_of_bounds("second argument of array.swap: index 6 not in range [0, 3]")
% ERROR-6-NOT:{{.}}
% ERROR-7:Uncaught Mercury exception:
% ERROR-7-NEXT:Software Error: predicate `array.fill_range'/5: Unexpected: empty range
% ERROR-7-NOT:{{.}}
% ERROR-8:Uncaught Mercury exception:
% ERROR-8-NEXT:index_out_of_bounds("second argument of fill_range: index -1 not in range [0, 3]")
% ERROR-8-NOT:{{.}}
% ERROR-9:Uncaught Mercury exception:
% ERROR-9-NEXT:index_out_of_bounds("third argument of fill_range: index 9 not in range [0, 3]")
% ERROR-9-NOT:{{.}}
% ERROR-10:Uncaught Mercury exception:
% ERROR-10-NEXT:Software Error: predicate `array.init'/3: Unexpected: negative size
% ERROR-10-NOT:{{.}}
% ERROR-11:Uncaught Mercury exception:
% ERROR-11-NEXT:Software Error: predicate `array.resize'/4: Unexpected: cannot resize to a negative size
% ERROR-11-NOT:{{.}}
% ERROR-12:Uncaught Mercury exception:
% ERROR-12-NEXT:Software Error: predicate `array.shrink'/3: Unexpected: cannot shrink to a larger size
% ERROR-12-NOT:{{.}}
% ERROR-13:Uncaught Mercury exception:
% ERROR-13-NEXT:Software Error: predicate `array.shrink'/3: Unexpected: cannot shrink to a negative size
% ERROR-13-NOT:{{.}}
% ERROR-14:Uncaught Mercury exception:
% ERROR-14-NEXT:index_out_of_bounds("array.set: index 14 not in range [0, 3]")
% ERROR-14-NOT:{{.}}
% ERROR-15:Uncaught Mercury exception:
% ERROR-15-NEXT:index_out_of_bounds("array.lookup: index 0 not in range [0, -1]")
% ERROR-15-NOT:{{.}}
% ERROR-16:Uncaught Mercury exception:
% ERROR-16-NEXT:Software Error: function `array.generate'/2: Unexpected: negative size
% ERROR-16-NOT:{{.}}
% ERROR-17:Uncaught Mercury exception:
% ERROR-17-NEXT:Software Error: predicate `array.generate_foldl'/5: Unexpected: negative size
% ERROR-17-NOT:{{.}}
% ERROR-18:Uncaught Mercury exception:
% ERROR-18-NEXT:Software Error: predicate `array.generate_foldl2'/7: Unexpected: negative size
% ERROR-18-NOT:{{.}}

:- module array_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.
:- pred which(int::out) is det.
:- pragma no_inline(which/1).
which(0).
main(!IO) :-
    which(W),
    A0 = array.init(4, 7),
    io.write_string("before\n", !IO),
    ( if W = 1 then
        io.write_int(array.lookup(A0, 4), !IO)
    else if W = 2 then
        io.write_int(A0 ^ elem(-1), !IO)
    else if W = 3 then
        array.set(W + 1, 1, A0, A), io.write_int(A ^ elem(0), !IO)
    else if W = 4 then
        A = A0 ^ elem(9) := 1, io.write_int(A ^ elem(0), !IO)
    else if W = 5 then
        array.swap(W, 0, A0, A), io.write_int(A ^ elem(0), !IO)
    else if W = 6 then
        array.swap(0, W, A0, A), io.write_int(A ^ elem(0), !IO)
    else if W = 7 then
        array.fill_range(1, W, 3, A0, A), io.write_int(A ^ elem(0), !IO)
    else if W = 8 then
        array.fill_range(1, -1, 3, A0, A), io.write_int(A ^ elem(0), !IO)
    else if W = 9 then
        array.fill_range(1, 0, W, A0, A), io.write_int(A ^ elem(0), !IO)
    else if W = 10 then
        A = array.init(-W, 1), io.write_int(array.size(A), !IO)
    else if W = 11 then
        array.resize(-W, 1, A0, A), io.write_int(array.size(A), !IO)
    else if W = 12 then
        array.shrink(W, A0, A), io.write_int(array.size(A), !IO)
    else if W = 13 then
        A = array.shrink(A0, -W), io.write_int(array.size(A), !IO)
    else if W = 14 then
        A = array.slow_set(A0, W, 1), io.write_int(A ^ elem(0), !IO)
    else if W = 15 then
        A = array.make_empty_array : array(int), io.write_int(array.lookup(A, 0), !IO)
    else if W = 16 then
        A = array.generate(-W, double), io.write_int(array.size(A), !IO)
    else if W = 17 then
        array.generate_foldl(-W, count, A, 0, C), io.write_int(array.size(A) + C, !IO)
    else if W = 18 then
        array.generate_foldl2(-W, count2, A, 0, C, 0, D),
        io.write_int(array.size(A) + C + D, !IO)
    else
        io.write_int(array.size(A0), !IO)
    ),
    io.write_string("after\n", !IO).

:- func double(int) = int.
double(I) = 2 * I.

:- pred count(int::in, int::out, int::in, int::out) is det.
count(I, I, N, N + 1).

:- pred count2(int::in, int::out, int::in, int::out, int::in, int::out)
    is det.
count2(I, I, N, N + 1, M, M - 1).
