% MANUAL: Data-functors
% Character literals, quoted or unquoted and an operator in parentheses,
% and the implementation-defined literals $line, $module and $pred.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:['a', 'B', '+', '.']
% OUTPUT-NEXT:{"run_literals_and_characters", "predicate `run_literals_and_characters.where\'/3", 23}
% OUTPUT-NOT:{{.}}

:- module run_literals_and_characters.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred where(string::out, string::out, int::out) is det.
where($module, $pred, $line).

main(!IO) :-
    io.write([a, 'B', (+), ('.')], !IO), io.nl(!IO),
    where(M, P, L),
    io.write({M, P, L}, !IO), io.nl(!IO).
