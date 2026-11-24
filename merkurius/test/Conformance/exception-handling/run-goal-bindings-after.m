% MANUAL: Exception handling
% The manual says the variables GOAL binds are visible in THENGOAL only;
% mmc lets the goals after the try goal see them where no catch can succeed
% without binding them, as here with none, and so does merkc (design
% document, "Where mmc and the manual disagree").
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:then 11 d
% OUTPUT-NEXT:after 11 d
% OUTPUT-NEXT:after 20
% OUTPUT-NOT:{{.}}

:- module run_goal_bindings_after.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    A = 10,
    ( try [] ( C = A + 1, D = "d" )
    then
        io.format("then %d %s\n", [i(C), s(D)], !IO)
    ),
    io.format("after %d %s\n", [i(C), s(D)], !IO),
    ( try [] E = A * 2
    then
        true
    catch "x" ->
        throw("y")
    ),
    io.format("after %d\n", [i(E)], !IO).
