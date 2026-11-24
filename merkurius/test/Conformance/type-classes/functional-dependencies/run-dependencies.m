% MANUAL: Functional dependencies
% Functional dependencies: a method that uses only the domain parameter,
% the manual's mutually recursive dependencies, an instance whose range
% is determined through its constraint, a predicate whose constraint's
% variable is determined only by the dependency.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:"int"
% OUTPUT-NEXT:"list of list of int"
% OUTPUT-NEXT:["4", "5"]
% OUTPUT-NEXT:"p"
% OUTPUT-NOT:{{.}}

:- module run_dependencies.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, string.

    % In place of map(K, V), which the manual's example uses.
:- type table(K, V) ---> table(list(K), list(V)).

:- typeclass baz(A, B) <= (A -> B) where [
    func name(A) = string,
    func default(A) = B
].

:- instance baz(int, string) where [
    name(_) = "int",
    default(N) = string.int_to_string(N)
].

:- instance baz(list(S), list(T)) <= baz(S, T) where [
    name(L) = "list of " ++ ( if L = [X | _] then name(X) else "nothing" ),
    default(L) = list.map((func(X) = default(X)), L)
].

:- typeclass foo(A, B) <= ((A -> B), (B -> A)) where [].
:- typeclass bar(A, B, C, D) <= ((A, B -> C), (B, C -> D), (D -> A, C))
    where [].

:- pred p(X, Y, string) <= baz(table(X, Y), list(Z)).
:- mode p(in, in, out) is det.
p(_, _, "p").

:- instance baz(table(K, V), list(K)) where [
    name(_) = "table",
    default(table(Ks, _)) = Ks
].

main(!IO) :-
    io.write_line(name(3), !IO),
    io.write_line(name([[1, 2]]), !IO),
    D = default([4, 5]),
    io.write_line(D, !IO),
    p(1, "one", S),
    io.write_line(S, !IO).
