% MANUAL: Exception handling
% The library's errors are exceptions a try catches: error/1's
% software_error and the domain_error of a division by zero.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:3
% OUTPUT-NEXT:domain_error: int.'//': division by zero
% OUTPUT-NEXT:software_error: one
% OUTPUT-NOT:{{.}}

:- module run_library_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, require, string.

:- pred divide(int::in, int::in, io::di, io::uo) is cc_multi.
divide(X, Y, !IO) :-
    ( try [] ( if Y = 1 then error("one") else Q = X // Y )
    then
        io.format("%d\n", [i(Q)], !IO)
    catch domain_error(M) ->
        io.write_string("domain_error: " ++ M ++ "\n", !IO)
    catch software_error(M) ->
        io.write_string("software_error: " ++ M ++ "\n", !IO)
    ).

main(!IO) :-
    divide(12, 4, !IO),
    divide(12, 0, !IO),
    divide(12, 1, !IO).
