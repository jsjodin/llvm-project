% MANUAL: Exception handling
% DIFFERS: mmc-bug: mmc aborts on an else part of a det goal whose then
% part passes the goal's binding in a list ("Unexpected: error merging var
% 6" in merge_instmapping_delta_vars). merkc runs it, never running the
% else part.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:then 1
% OUTPUT-NOT:{{.}}
% MMC-STATUS: rejected
% MMC:

:- module differs_else_det_format.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] X = 1
    then
        io.format("then %d\n", [i(X)], !IO)
    else
        io.write_string("else\n", !IO)
    ).
