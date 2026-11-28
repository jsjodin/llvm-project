% MANUAL: Explicit type qualification
% DESIGN: Sized floating point types
% Term : Type, with_type and `with_type` resolve an ambiguity, with a type
% variable of the predicate's declaration in scope. It is a template, run
% for float and float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_type_qualification_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_type_qualification_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:{[], [], [], 2}
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_type_qualification_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_type_qualification_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:{[], [], [], 2}
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_type_qualification.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, string.

:- pred len(list(T)::in, int::out) is det.
len(L, N) :- L2 = L : list(T), N = list.length(L2).

main(!IO) :-
    A = [] : list(int),
    B = with_type([], list(string)),
    C = [] `with_type` list(@T@),
    len([1, 2], N),
    io.write({A, B, C, N}, !IO), io.nl(!IO).
