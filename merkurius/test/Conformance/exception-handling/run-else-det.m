% MANUAL: Exception handling
% The manual says the else part must be omitted if GOAL cannot fail; mmc
% accepts one, which never runs, and so does merkc (design document, "Where
% mmc and the manual disagree").
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1
% OUTPUT-NOT:{{.}}

:- module run_else_det.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] X = 1
    then
        io.write_int(X, !IO),
        io.nl(!IO)
    else
        io.write_string("else\n", !IO)
    ).
