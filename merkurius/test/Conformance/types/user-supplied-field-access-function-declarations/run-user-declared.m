% MANUAL: User-supplied field access function declarations
% Declarations of a field's access functions replace the generated ones,
% as an elem with an extra argument does, like map's, in field syntax in the
% declarations and clause heads. A field access function may also be
% defined for a field that is no type's, computed each time.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1
% OUTPUT-NEXT:point(5, 2)
% OUTPUT-NEXT:7
% OUTPUT-NEXT:point(5, 15)
% OUTPUT-NEXT:5
% OUTPUT-NEXT:2
% OUTPUT-NEXT:missing
% OUTPUT-NEXT:table(["b" - 2, "a" - 1])
% OUTPUT-NOT:{{.}}

:- module run_user_declared.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, pair.

:- type point
    --->    point(x :: int, y :: int).

:- func point ^ x = int.
:- mode in ^ x = out is det.

:- func (point ^ x := int) = point.
:- mode (in ^ x := in) = out is det.

    % A field no type has.
:- func point ^ sum = int.
sum(P) = P ^ x + P ^ y.

:- func (point ^ sum := int) = point.
(P ^ sum := S) = P ^ y := S - P ^ x.

    % A table with map's elem functions (merkc's library has no map yet).
:- type table(K, V)
    --->    table(list(pair(K, V))).

:- func table(K, V) ^ elem(K) = V.
:- mode in ^ elem(in) = out is semidet.
table(Pairs) ^ elem(Key) = Value :-
    search(Pairs, Key, Value).

:- pred search(list(pair(K, V))::in, K::in, V::out) is semidet.
search([K - V | Pairs], Key, Value) :-
    ( if K = Key then
        Value = V
    else
        search(Pairs, Key, Value)
    ).

:- func (table(K, V) ^ elem(K) := V) = table(K, V).
:- mode (in ^ elem(in) := in) = out is det.
(table(Pairs) ^ elem(Key) := Value) = table([Key - Value | Pairs]).

main(!IO) :-
    P0 = point(1, 2),
    io.write_line(P0 ^ x, !IO),
    P1 = P0 ^ x := 5,
    io.write_line(P1, !IO),
    io.write_line(P1 ^ sum, !IO),
    P2 = P1 ^ sum := 20,
    io.write_line(P2, !IO),
    GetX = x,
    io.write_line(GetX(P2), !IO),
    M0 = table([]) : table(string, int),
    M1 = M0 ^ elem("a") := 1,
    M2 = M1 ^ elem("b") := 2,
    ( if V = M2 ^ elem("b") then
        io.write_line(V, !IO)
    else
        io.write_string("missing\n", !IO)
    ),
    ( if W = M2 ^ elem("c") then
        io.write_line(W, !IO)
    else
        io.write_string("missing\n", !IO)
    ),
    io.write_line(M2, !IO).
