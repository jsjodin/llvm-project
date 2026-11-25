% MANUAL: Exception handling
% The io try parameter threads the I/O state through GOAL. The manual's
% example, p_carefully, with a p that returns, one that throws a string, one
% that throws 42, and one that throws another value, which p_carefully
% rethrows.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:Calling p
% OUTPUT-NEXT:p returned: 0
% OUTPUT-NEXT:Calling p
% OUTPUT-NEXT:p threw a string: one
% OUTPUT-NEXT:Calling p
% OUTPUT-NEXT:p threw 42
% OUTPUT-NEXT:Calling p
% OUTPUT-NEXT:p threw something: 3.5
% OUTPUT-NEXT:main caught 3.5
% OUTPUT-NOT:{{.}}

:- module run_io.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

:- pred p(int::in, int::out, io::di, io::uo) is det.
p(N, Output, !IO) :-
    ( if N = 1 then
        throw("one")
    else if N = 2 then
        throw(42)
    else if N = 3 then
        throw(3.5)
    else
        Output = N * 10
    ).

:- pred p_carefully(int::in, io::di, io::uo) is cc_multi.
p_carefully(N, !IO) :-
    (try [io(!IO)] (
        io.write_string("Calling p\n", !IO),
        p(N, Output, !IO)
    )
    then
        io.write_string("p returned: ", !IO),
        io.write(Output, !IO),
        io.nl(!IO)
    catch S ->
        io.write_string("p threw a string: ", !IO),
        io.write_string(S, !IO),
        io.nl(!IO)
    catch 42 ->
        io.write_string("p threw 42\n", !IO)
    catch_any Other ->
        io.write_string("p threw something: ", !IO),
        io.write(Other, !IO),
        io.nl(!IO),
        % Rethrow the value.
        throw(Other)
    ).

main(!IO) :-
    p_carefully(0, !IO),
    p_carefully(1, !IO),
    p_carefully(2, !IO),
    ( try [io(!IO)] p_carefully(3, !IO)
    then
        true
    catch F ->
        io.format("main caught %.1f\n", [f(F)], !IO)
    ).
