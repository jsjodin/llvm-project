% A division by zero throws the library's exception, which nothing catches
% yet: the program flushes its output, writes the exception to the
% standard error as mmc's programs do, and exits with status 1. zero/1's
% no_inline keeps the divisor from being a constant.
%
% RUN: merkc %s -o %t
% RUN: not %t > %t.out 2> %t.err
% RUN: FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace < %t.out
% RUN: FileCheck %s --check-prefix=ERROR --match-full-lines --strict-whitespace < %t.err

% OUTPUT:before
% OUTPUT-NOT:{{.}}

% ERROR:Uncaught Mercury exception:
% ERROR-NEXT:domain_error("int.\'//\': division by zero")
% ERROR-NOT:{{.}}

:- module arithmetic_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.
:- pred zero(int::out) is det.
:- pragma no_inline(zero/1).
zero(0).
main(!IO) :-
    zero(Z),
    io.write_string("before\n", !IO),
    io.write_int(7 // Z, !IO),
    io.write_string("after\n", !IO).
