% MANUAL: Exception handling
% A try goal in a polymorphic predicate with a type class constraint, whose
% goal throws a value of the type variable's type.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:area 9
% OUTPUT-NEXT:threw square(-1)
% OUTPUT-NOT:{{.}}

:- module run_polymorphic.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

:- typeclass shape(T) where [
    func area(T) = int
].

:- type square
    --->    square(int).

:- instance shape(square) where [
    area(square(N)) = N * N
].

:- pred show(T::in, io::di, io::uo) is cc_multi <= shape(T).
show(Shape, !IO) :-
    ( try [] ( A = area(Shape), ( if A = 1 then throw(Shape) else true ) )
    then
        io.format("area %d\n", [i(A)], !IO)
    catch_any E ->
        io.write_string("threw ", !IO),
        io.write(E, !IO),
        io.nl(!IO)
    ).

main(!IO) :-
    show(square(3), !IO),
    show(square(-1), !IO).
