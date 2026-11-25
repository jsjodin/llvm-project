% MANUAL: Terms
% The equivalent forms of a list term.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{{\[\[}}1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
% OUTPUT-NEXT:same
% OUTPUT-NOT:{{.}}

:- module run_lists.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    L = [[1, 2, 3], [1, 2, 3 | []], [1, 2 | [3]], [1 | [2, 3]],
        '[|]'(1, '[|]'(2, '[|]'(3, [])))],
    io.write(L, !IO), io.nl(!IO),
    ( if L = [A | As], list.all_true(unify(A), As) then io.write_string("same\n", !IO) else true ).
