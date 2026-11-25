% MANUAL: Implicit quantification
% A variable that occurs only in a negated context, or a condition and
% its then part, is quantified there, so the same name in parallel goals
% names different variables.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:none > 5
% OUTPUT-NEXT:0
% OUTPUT-NEXT:8
% OUTPUT-NOT:{{.}}

:- module run_negated_context.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    L = [1, 2, 3],
    ( if not (list.member(X, L), X > 5) then io.write_string("none > 5", !IO) else true ),
    io.nl(!IO),
    ( if X = 2, list.member(X, L) then io.write_int(X, !IO) else io.write_int(0, !IO) ),
    io.nl(!IO),
    ( if 1 > 2 then X = 7 else X = 8 ), io.write_int(X, !IO), io.nl(!IO).
