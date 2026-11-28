% MANUAL: Syntax and semantics of type specialization pragmas
% DESIGN: Sized floating point types
% type_spec pragmas for all of a predicate's modes and for one, with one
% variable and two, and with a new variable on the right, which do not
% change what a program does. It is a template, run for float and float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_type_spec_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_type_spec_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:b70.5
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_type_spec_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_type_spec_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:b70.5
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_type_spec.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, string.

:- pred lookup(list({K, V})::in, K::in, V::out) is semidet.
:- pragma type_spec(lookup/3, K = int).
:- pragma type_spec(lookup(in, in, out), (K = string, V = int)).
:- pragma type_spec(lookup/3, K = list(U)).
lookup([{K0, V0} | T], K, V) :-
    ( if K = K0 then V = V0 else lookup(T, K, V) ).

main(!IO) :-
    ( if lookup([{1, "a"}, {2, "b"}], 2, V) then io.write_string(V, !IO) else true ),
    ( if lookup([{"x", 7}], "x", W) then io.write_int(W, !IO) else true ),
    ( if lookup([{[1], 0.5@S@}], [1], F) then io.write_@T@(F, !IO) else true ),
    io.nl(!IO).
