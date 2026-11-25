% MANUAL: Builtin operators
% Priorities and associativities of the arithmetic, bitwise, string and
% range operators.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[10, 14, 5, 6, 3, 3, 5, 10, -1, 1]
% OUTPUT-NEXT:{"abc", [1, 2, 3], [1, 2]}
% OUTPUT-NOT:{{.}}

:- module run_precedence.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    io.write([2 * 3 + 4, 2 + 3 * 4, 10 - 3 - 2, 7 // 2 * 2,
        7 rem 4, 7 mod 4, 1 << 2 + 1, 6 /\ 3 \/ 8, \ 0, - 2 + 3], !IO),
    io.nl(!IO),
    io.write({"a" ++ "b" ++ "c", 1 .. 3, [1] ++ [2]}, !IO), io.nl(!IO).
