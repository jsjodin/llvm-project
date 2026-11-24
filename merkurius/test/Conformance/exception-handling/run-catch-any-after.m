% MANUAL: Exception handling
% The manual says CATCHANYVAR is visible in CATCHANYGOAL only; mmc lets the
% goals after the try goal see it where GOAL cannot succeed, and so does
% merkc (design document, "Where mmc and the manual disagree").
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:caught
% OUTPUT-NEXT:5
% OUTPUT-NOT:{{.}}

:- module run_catch_any_after.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] throw(5)
    then
        io.write_string("then\n", !IO)
    catch_any E ->
        io.write_string("caught\n", !IO)
    ),
    io.write(E, !IO),
    io.nl(!IO).
