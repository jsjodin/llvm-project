% MANUAL: Terms
% Backquoted names, module-qualified names and variables as operators,
% in goals, clause heads and data-terms; left associative, binding more
% tightly than '*'.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:less
% OUTPUT-NEXT:[5, 8, -1, 9]
% OUTPUT-NOT:{{.}}

:- module run_operator_terms.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred less(int::in, int::in) is semidet.
X `less` Y :- X < Y.
:- func sub(int, int) = int.
X `sub` Y = X - Y.

main(!IO) :-
    ( if 1 `less` 2 then io.write_string("less\n", !IO) else true ),
    F = sub,
    io.write([10 `sub` 3 `sub` 2, 2 * 5 `sub` 1, 3 `F` 4, 3 `int.max` 9], !IO),
    io.nl(!IO).
