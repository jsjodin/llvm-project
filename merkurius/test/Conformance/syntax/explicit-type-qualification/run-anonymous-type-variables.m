% MANUAL: Explicit type qualification
% An anonymous type variable in a qualification, _, is a variable of its
% own at each occurrence, as _ is in a term, so two qualifications with
% list(_) may have different element types; a named one, _T, is the same
% throughout the clause. Each _ in a declaration is its own too.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[2, 1]
% OUTPUT-NEXT:["b", "a"]
% OUTPUT-NEXT:[2, 1]
% OUTPUT-NEXT:[1]
% OUTPUT-NEXT:["a"]
% OUTPUT-NOT:{{.}}
:- module run_anonymous_type_variables.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- pred both(list(_)::in, list(_)::in, io::di, io::uo) is det.

both(A, B, !IO) :-
    io.write_line(A, !IO),
    io.write_line(B, !IO).

main(!IO) :-
    L = [1, 2],
    M = ["a", "b"],
    io.write_line(list.reverse(L) : list(_), !IO),
    io.write_line(list.reverse(M) : list(_), !IO),
    io.write_line(list.reverse(L) : list(_T), !IO),
    both([1], ["a"], !IO).
