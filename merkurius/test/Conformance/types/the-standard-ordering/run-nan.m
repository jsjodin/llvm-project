% MANUAL: The standard ordering
% Comparing a float that is a NaN, which the manual says throws an
% exception, gives (=) as with mmc (design document, "Where mmc and the
% manual disagree", "Comparing a NaN").
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:'='
% OUTPUT-NOT:{{.}}

:- module run_nan.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, float.

main(!IO) :-
    NaN = float.infinity - float.infinity,
    ( try []
        compare(R, NaN, 1.0)
    then
        io.write_line(R, !IO)
    catch_any E ->
        io.write_string("exception: ", !IO),
        io.write_line(E, !IO)
    ).
