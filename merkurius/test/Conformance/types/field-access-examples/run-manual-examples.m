% MANUAL: Field access examples
% The manual's examples: the generated functions of type1's fields, the
% nested update increment_field3, and a field of a table's element, as the
% manual's of a map's.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:type2(1, 2)
% OUTPUT-NEXT:2
% OUTPUT-NEXT:type1(type2(1, 2), "three")
% OUTPUT-NEXT:type1(type2(2, 2), "two")
% OUTPUT-NEXT:type1(type2(4, 2), "two")
% OUTPUT-NEXT:table([2 - type1(type2(2, 2), "updated"), 1 - type1(type2(1, 2), "two"), 2 - type1(type2(2, 2), "two")])
% OUTPUT-NEXT:no index 3
% OUTPUT-NOT:{{.}}

:- module run_manual_examples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, pair.

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

:- type type1
    --->    type1(
                field1 :: type2,
                field2 :: string
            ).

:- type type2
    --->    type2(
                field3 :: int,
                field4 :: int
            ).

:- func type1 ^ increment_field3 = type1.
Term0 ^ increment_field3 =
    Term0 ^ field1 ^ field3 := Term0 ^ field1 ^ field3 + 1.

:- func update_field_in_map(table(int, type1), int, string)
    = table(int, type1) is semidet.
update_field_in_map(Map, Index, Value) =
    Map ^ elem(Index) ^ field2 := Value.

main(!IO) :-
    T0 = type1(type2(1, 2), "two"),
    io.write_line(T0 ^ field1, !IO),
    io.write_line(T0 ^ field1 ^ field4, !IO),
    io.write_line(T0 ^ field2 := "three", !IO),
    T1 = T0 ^ increment_field3,
    io.write_line(T1, !IO),
    io.write_line(T1 ^ increment_field3 ^ increment_field3, !IO),
    M0 = table([1 - T0, 2 - T1]),
    ( if M1 = update_field_in_map(M0, 2, "updated") then
        io.write_line(M1, !IO)
    else
        io.write_string("no index 2\n", !IO)
    ),
    ( if M2 = update_field_in_map(M0, 3, "updated") then
        io.write_line(M2, !IO)
    else
        io.write_string("no index 3\n", !IO)
    ).
