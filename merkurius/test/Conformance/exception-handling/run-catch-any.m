% MANUAL: Exception handling
% If no catch's term unifies with the exception, the catch_any branch runs
% with the exception bound to its variable, which has type some [T] T.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:any: point(1, 2)
% OUTPUT-NEXT:any: 2.5
% OUTPUT-NEXT:the string
% OUTPUT-NEXT:any: "other"
% OUTPUT-NOT:{{.}}

:- module run_catch_any.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

:- type point
    --->    point(int, int).

:- pred show(T::in, io::di, io::uo) is cc_multi.
show(V, !IO) :-
    ( try [] throw(V)
    then
        io.write_string("then\n", !IO)
    catch "string" ->
        io.write_string("the string\n", !IO)
    catch_any E ->
        io.write_string("any: ", !IO),
        io.write(E, !IO),
        io.nl(!IO)
    ).

main(!IO) :-
    show(point(1, 2), !IO),
    show(2.5, !IO),
    show("string", !IO),
    show("other", !IO).
