% MANUAL: Discriminated unions
% Curly braces quote a semicolon in a type's body, and then must quote
% braces too: the manual's type tricky has the constructors ';'/2 and
% '{}'/1.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1 ; 2
% OUTPUT-NEXT:{ 3 }
% OUTPUT-NEXT:{1, 2}
% OUTPUT-NEXT:3
% OUTPUT-NOT:{{.}}

:- module run_quoted_semicolon.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type tricky
    --->    { int ; int }
    ;       { { int } }.

main(!IO) :-
    A = (1 ; 2) : tricky,
    B = {3} : tricky,
    io.write_line(A, !IO),
    io.write_line(B, !IO),
    ( if A = (X ; Y) then
        io.write_line({X, Y}, !IO)
    else
        true
    ),
    ( if B = {Z} then
        io.write_line(Z, !IO)
    else
        true
    ).
