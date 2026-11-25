% MANUAL: Exception handling
% Variables bound by the unification of a catch's term are in scope of its
% goal.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:bad input 3: three
% OUTPUT-NOT:{{.}}

:- module run_catch_binds.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

:- type error_kind
    --->    bad_input(int, string)
    ;       other.

main(!IO) :-
    ( try [] throw(bad_input(3, "three"))
    then
        io.write_string("then\n", !IO)
    catch other ->
        io.write_string("other\n", !IO)
    catch bad_input(N, Why) ->
        io.format("bad input %d: %s\n", [i(N), s(Why)], !IO)
    ).
