% MANUAL: The standard ordering
% DIFFERS: manual: comparing a NaN that a type variable stands for throws
% an exception, as the manual says, which a try catches: a software_error.
% mmc's program aborts ("Mercury runtime: incomparable floats in
% compare/3"). See "Comparing a NaN" in the design document's register.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:exception: software_error("incomparable floats in compare/3")
% OUTPUT-NOT:{{.}}
% MMC-STATUS: 1
% MMC: Mercury runtime: incomparable floats in compare/3

:- module differs_nan_type_variable.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, float.

:- type g(T)
    --->    g1
    ;       g2(T).

main(!IO) :-
    NaN = float.infinity - float.infinity,
    ( try []
        compare(R, g2(NaN), g2(1.0))
    then
        io.write_line(R, !IO)
    catch_any E ->
        io.write_string("exception: ", !IO),
        io.write_line(E, !IO)
    ).
