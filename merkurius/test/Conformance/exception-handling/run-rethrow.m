% MANUAL: Exception handling
% If the thrown value unifies with no catch's term, and there is no
% catch_any branch, the exception is rethrown, here to an enclosing try.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:outer caught 1
% OUTPUT-NOT:{{.}}

:- module run_rethrow.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] (
        try [] throw(1)
        then true
        catch "x" -> true
      )
    then
        io.write_string("then\n", !IO)
    catch N ->
        io.format("outer caught %d\n", [i(N)], !IO)
    ).
