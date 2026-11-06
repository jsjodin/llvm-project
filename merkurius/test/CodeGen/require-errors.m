% The library's require module: each error throws software_error with mmc's
% message, which ends the program as an uncaught exception, and require and
% expect throw only when their goal fails, expect_not when it succeeds. Each
% case, chosen by which/1, prints a line first; the output is mmc's.
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

:- module require_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, require.

:- pred which(int::out) is det.
which(0).

main(!IO) :-
    io.write_string("before\n", !IO),
    which(N),
    run(N, !IO).

:- pred fails is semidet.
fails :- which(N), N > 100.
:- pred succeeds is semidet.
succeeds :- which(N), N < 100.

:- pred run(int::in, io::di, io::uo) is det.
run(N, !IO) :-
    ( if N = 0 then
        require(succeeds, "not thrown"),
        expect(succeeds, "mod", "not thrown"),
        expect_not(fails, "mod", "not thrown"),
        io.write_string("after\n", !IO)
    else if N = 1 then
        error("boom")
    else if N = 2 then
        error("pred p/1", "boom")
    else if N = 3 then
        io.write_int(func_error("pred p/1", "boom") : int, !IO)
    else if N = 4 then
        sorry("mod", "what")
    else if N = 5 then
        io.write_int(sorry("mod", "pred p/1", "what") : int, !IO)
    else if N = 6 then
        unexpected("mod", "what")
    else if N = 7 then
        unexpected("mod", "pred p/1", "what")
    else if N = 8 then
        require(fails, "req msg")
    else if N = 9 then
        expect(fails, "mod", "pred p/1", "expect msg")
    else if N = 10 then
        expect_not(succeeds, "mod", "not msg")
    else if N = 11 then
        error("line1\nline2")
    else
        io.write_string("none\n", !IO)
    ).

% OUTPUT-0:before
% OUTPUT-0-NEXT:after
% OUTPUT:before
% OUTPUT-EMPTY:
% ERROR-1:Uncaught Mercury exception:
% ERROR-1-NEXT:Software Error: boom
% ERROR-2:Uncaught Mercury exception:
% ERROR-2-NEXT:Software Error: pred p/1: boom
% ERROR-3:Uncaught Mercury exception:
% ERROR-3-NEXT:Software Error: pred p/1: boom
% ERROR-4:Uncaught Mercury exception:
% ERROR-4-NEXT:Software Error: mod: Sorry, not implemented: what
% ERROR-5:Uncaught Mercury exception:
% ERROR-5-NEXT:Software Error: mod: pred p/1: Sorry, not implemented: what
% ERROR-6:Uncaught Mercury exception:
% ERROR-6-NEXT:Software Error: mod: Unexpected: what
% ERROR-7:Uncaught Mercury exception:
% ERROR-7-NEXT:Software Error: mod: pred p/1: Unexpected: what
% ERROR-8:Uncaught Mercury exception:
% ERROR-8-NEXT:Software Error: req msg
% ERROR-9:Uncaught Mercury exception:
% ERROR-9-NEXT:Software Error: mod: pred p/1: Unexpected: expect msg
% ERROR-10:Uncaught Mercury exception:
% ERROR-10-NEXT:Software Error: mod: Unexpected: not msg
% ERROR-11:Uncaught Mercury exception:
% ERROR-11-NEXT:Software Error: line1
% ERROR-11-NEXT:line2
