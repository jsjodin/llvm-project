% MANUAL: User-defined equality and comparison
% An equality predicate's type, mode and determinism may be more
% permissive: here its type is the polymorphic pred(T, T) and its
% determinism det, so all values of the type are equal; a comparison
% predicate's type may be polymorphic too.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:equal
% OUTPUT-NEXT:'='
% OUTPUT-NOT:{{.}}

:- module run_permissive_signature.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type token
    --->    token(int)
            where equality is always_equal, comparison is always_same.

:- pred always_equal(T::in, T::in) is det.
always_equal(_, _).

:- pred always_same(comparison_result::uo, T::in, T::in) is det.
always_same((=), _, _).

:- pred test(token::in, token::in, io::di, io::uo) is det.
test(A, B, !IO) :-
    ( if A = B then
        io.write_string("equal\n", !IO)
    else
        io.write_string("different\n", !IO)
    ),
    compare(R, A, B),
    io.write_line(R, !IO).

main(!IO) :-
    test(token(1), token(2), !IO).
