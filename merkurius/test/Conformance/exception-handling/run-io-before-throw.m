% MANUAL: Exception handling
% I/O that GOAL performs before it throws is done, and the I/O state
% continues in the catch.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:before
% OUTPUT-NEXT:caught 42
% OUTPUT-NEXT:after
% OUTPUT-NOT:{{.}}

:- module run_io_before_throw.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [io(!IO)] (
        io.write_string("before\n", !IO),
        throw(42),
        io.write_string("not reached\n", !IO)
    )
    then
        io.write_string("then\n", !IO)
    catch N ->
        io.format("caught %d\n", [i(N)], !IO)
    ),
    io.write_string("after\n", !IO).
