% MANUAL: Record syntax
% Field selection and update, of nested fields and of a field with an
% argument, as the manual transforms them.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{outer(inner(5), "b"), 5, 8}
% OUTPUT-NOT:{{.}}

:- module run_record_syntax.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type inner ---> inner(v :: int).
:- type outer ---> outer(i :: inner, name :: string).
:- func at(int, list(int)) = int.
at(N, L) = list.det_index0(L, N).

main(!IO) :-
    O0 = outer(inner(1), "a"),
    O1 = O0 ^ i ^ v := 5,
    O2 = (O1 ^ name := "b"),
    io.write({O2, O2 ^ i ^ v, [7, 8, 9] ^ at(1)}, !IO), io.nl(!IO).
