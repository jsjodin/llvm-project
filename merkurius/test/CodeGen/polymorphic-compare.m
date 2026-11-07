% The standard ordering (reference manual, "The standard ordering"), at known
% types and through a type_info's descriptor: each kind of value, constructors
% in the order declared, arguments from the first, arrays by size first, and
% the operators on it. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module polymorphic_compare.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int8, list, uint, uint8.

:- type colour ---> red ; green ; blue.
% Neither the alphabet nor the layout gives the order.
:- type rev ---> zz ; yy ; xx.
:- type t ---> k(int, string) ; aa ; m(t) ; bb ; e(float, colour).
:- type wrap ---> wrap(int).
:- type unit ---> unit.
:- type pt(A, B) ---> pt(A, B) ; none.

% MERK-DAG: merk.type_ctor "int/0" module "" name "int" kind "int" linkage "linkonce" arity 0 unify @"unify_entry.int/0" compare @"compare_entry.int/0"
% LLVM-DAG: @"merk.type_ctor.int/0" = linkonce_odr constant { i8, i8, i16, i32, ptr, ptr, ptr, ptr, ptr, [8 x i16], ptr, ptr } { i8 0, i8 0, i16 0, i32 0, ptr @{{.*}}, ptr @{{.*}}, ptr @"unify_entry.int/0", ptr @"compare_entry.int/0", ptr null, [8 x i16] [i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1], ptr null, ptr null }

:- pred p(comparison_result::in, io::di, io::uo) is det.
p((=), !IO) :- io.write_string("=", !IO).
p((<), !IO) :- io.write_string("<", !IO).
p((>), !IO) :- io.write_string(">", !IO).

% MERK-DAG: merk.compare_word %arg0, %arg1, %arg2 : <"builtin.comparison_result">
:- pred c(T::in, T::in, io::di, io::uo) is det.
c(X, Y, !IO) :-
    compare(R, X, Y),
    p(R, !IO).

% Each pair at T, both ways round and against itself.
:- pred each(T::in, T::in, io::di, io::uo) is det.
each(X, Y, !IO) :-
    c(X, Y, !IO),
    c(Y, X, !IO),
    c(X, X, !IO),
    io.nl(!IO).

:- pragma no_inline(c/4).
:- pragma no_inline(each/4).

main(!IO) :-
    % OUTPUT:<>=
    each(1, 2, !IO),
    % OUTPUT:<>=
    each(-1, 2, !IO),
    % OUTPUT:<>=
    each(1u, 2u, !IO),
    % OUTPUT:><=
    each(0xffffffffffffffffu, 2u, !IO),
    % OUTPUT:<>=
    each(-1i8, 1i8, !IO),
    % OUTPUT:><=
    each(255u8, 1u8, !IO),
    % OUTPUT:<>=
    each('p', 'q', !IO),
    % OUTPUT:><=
    each('é', 'z', !IO),
    % OUTPUT:><=
    each('😀', 'é', !IO),
    % OUTPUT:<>=
    each("a", "b", !IO),
    % OUTPUT:<>=
    each("", "a", !IO),
    % OUTPUT:><=
    each("ab", "a", !IO),
    % OUTPUT:><=
    each("é", "z", !IO),
    % OUTPUT:<>=
    each(1.0, 2.0, !IO),
    % OUTPUT:===
    each(0.0, -0.0, !IO),
    % OUTPUT:<>=
    each(red, blue, !IO),
    % OUTPUT:<>=
    each(zz, xx, !IO),
    % OUTPUT:<>=
    each(k(1, "a"), aa, !IO),
    % OUTPUT:<>=
    each(aa, bb, !IO),
    % OUTPUT:><=
    each(bb, m(aa), !IO),
    % OUTPUT:><=
    each(k(1, "b"), k(1, "a"), !IO),
    % OUTPUT:><=
    each(k(2, "a"), k(1, "b"), !IO),
    % OUTPUT:<>=
    each(m(k(1, "a")), m(aa), !IO),
    % OUTPUT:<>=
    each(e(1.0, red), e(1.0, blue), !IO),
    % OUTPUT:<>=
    each(wrap(1), wrap(2), !IO),
    % OUTPUT:===
    each(unit, unit, !IO),
    % OUTPUT:<>=
    each([1, 2], [1, 3], !IO),
    % OUTPUT:<>=
    each([], [1], !IO),
    % OUTPUT:><=
    each([1, 2], [1], !IO),
    % OUTPUT:<>=
    each(["a"], ["a", "b"], !IO),
    % OUTPUT:<>=
    each({1, "a"}, {1, "b"}, !IO),
    % OUTPUT:><=
    each({2, "a"}, {1, "b"}, !IO),
    % OUTPUT:<>=
    each(pt(1, "a"), none, !IO),
    % OUTPUT:<>=
    each(pt([1], red), pt([1], blue), !IO),

    % An array's size is more significant than its elements.
    array.init(3, 1, A3),
    array.init(1, 9, A1),
    array.init(2, 1, B1),
    array.init(2, 2, B2),
    array.init(0, 2, A0),
    % OUTPUT:><=
    each(A3, A1, !IO),
    % OUTPUT:<>=
    each(B1, B2, !IO),
    % OUTPUT:<>=
    each(A0, A1, !IO),
    % OUTPUT:<>=
    each([B1], [B2], !IO),

    % At known types, where no type_info is needed.
    % MERK-DAG: merk.proc private @compare.polymorphic_compare.t(%arg0: !merk.term<"polymorphic_compare.t">, %arg1: !merk.term<"polymorphic_compare.t">) -> !merk.term<"builtin.comparison_result">
    % MERK-DAG: merk.proc private @"compare.list.list(int)"(%arg0: !merk.term<"list.list(int)">,
    compare(R1, 1, 2), p(R1, !IO),
    compare(R2, 255u8, 1u8), p(R2, !IO),
    compare(R3, "b", "a"), p(R3, !IO),
    compare(R4, 1.0, 2.0), p(R4, !IO),
    compare(R5, red, blue), p(R5, !IO),
    compare(R6, m(aa), m(bb)), p(R6, !IO),
    compare(R7, [1, 2], [1, 3]), p(R7, !IO),
    compare(R8, {2, "a"}, {1, "b"}), p(R8, !IO),
    compare(R9, B2, B1), p(R9, !IO),
    compare(R10, unit, unit), p(R10, !IO),
    p(array_compare(B1, B2), !IO),
    io.nl(!IO),
    % OUTPUT:<>><<<<>>=<

    % The operators.
    ( if 1 @< 2 then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if 2 @< 2 then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if 2 @=< 2 then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if 3 @=< 2 then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if "b" @> "a" then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if "a" @> "a" then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if blue @>= blue then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if red @>= blue then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    p(ordering([2], [1]), !IO),
    ( if unify(1, 1) then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    % A bound result is tested.
    ( if compare((<), 1, 2) then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if compare((>), 1, 2) then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    io.nl(!IO),
    % OUTPUT:ynynynyn>yyn
    lt(1, 2, !IO),
    lt("b", "a", !IO),
    lt([1], [1], !IO),
    io.nl(!IO).
    % OUTPUT:yynn<nnyy>nyny=

:- pred lt(T::in, T::in, io::di, io::uo) is det.
lt(X, Y, !IO) :-
    ( if X @< Y then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if X @=< Y then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if X @> Y then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    ( if X @>= Y then io.write_string("y", !IO) else io.write_string("n", !IO) ),
    p(ordering(X, Y), !IO).
