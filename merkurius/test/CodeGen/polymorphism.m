% Polymorphic procedures compiled once for every type (design document,
% "Polymorphism"): a type_info comes first for each type variable, a value of
% a type variable is one word, and a caller converts its values to and from
% that word. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module polymorphism.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, float, int, int8, list, string, uint8.

:- type colour ---> red ; green ; blue.
:- type temp ---> temp(float).
:- type tree ---> leaf ; node(tree, int, tree).

% MERK-LABEL: merk.proc @polymorphism.main_2_0
% MERK: %[[INT:.*]] = merk.type_info "int/0"
% MERK: %[[W:.*]] = merk.to_word %{{.*}} : i64
% MERK: %[[R:.*]] = merk.call @polymorphism.id_2_0(%[[INT]], %[[W]])
% MERK: merk.from_word %[[R]] : i64
% LLVM-DAG: @"merk.type_ctor.int/0" = linkonce_odr constant { i8, i8, i16, i32, ptr, ptr, ptr, ptr } { i8 0, i8 0, i16 0, i32 0, ptr @{{.*}}, ptr @{{.*}}, ptr @"unify_entry.int/0", ptr @"compare_entry.int/0" }
% LLVM-DAG: @"merk.type_ctor.list.list/1" = linkonce_odr constant { i8, i8, i16, i32, ptr, ptr, ptr, ptr } { i8 18, i8 0, i16 1,

% MERK-LABEL: merk.proc private @polymorphism.id_2_0(%arg0: !merk.type_info, %arg1: !merk.word) -> !merk.word
% LLVM-LABEL: define {{.*}}i64 @polymorphism.id_2_0(ptr %0, i64 %1)
:- pred id(T::in, T::out) is det.
id(X, X).

% Two type variables, in order of first appearance, and a call that passes
% its own type_infos on.
% MERK-LABEL: merk.proc private @polymorphism.swap_4_0(%arg0: !merk.type_info, %arg1: !merk.type_info, %arg2: !merk.word, %arg3: !merk.word) -> (!merk.word, !merk.word)
% MERK: merk.call @polymorphism.id_2_0(%arg1, %arg3)
% MERK: merk.call @polymorphism.id_2_0(%arg0, %arg2)
:- pred swap(A::in, B::in, B::out, A::out) is det.
swap(A, B, B1, A1) :-
    id(B, B1),
    id(A, A1).

% A branch joins words like any other values.
:- pred twice(T::in, T::out) is det.
twice(X, Y) :-
    tagged(1, X, Y).

:- pred tagged(int::in, T::in, T::out) is det.
tagged(N, X, Y) :-
    ( if N > 0 then
        id(X, Y)
    else
        Y = X
    ).

:- func first(A, B) = A.
first(A, _) = A.

:- pred show_tree(tree::in, io::di, io::uo) is det.
show_tree(leaf, !IO) :-
    io.write_string("leaf", !IO).
show_tree(node(L, V, R), !IO) :-
    io.write_string("node(", !IO),
    show_tree(L, !IO),
    io.write_string(", ", !IO),
    io.write_int(V, !IO),
    io.write_string(", ", !IO),
    show_tree(R, !IO),
    io.write_string(")", !IO).

main(!IO) :-
    id(42, I),
    io.write_int(I, !IO),
    io.nl(!IO),
    id(-1.5, F),
    io.write_float(F, !IO),
    io.nl(!IO),
    id('x', C),
    io.write_char(C, !IO),
    io.nl(!IO),
    id(-3i8, N8),
    io.write_int8(N8, !IO),
    io.nl(!IO),
    id(200u8, U8),
    io.write_uint8(U8, !IO),
    io.nl(!IO),
    id("hello", S),
    io.write_string(S, !IO),
    io.nl(!IO),
    id(green, Colour),
    ( if Colour = green then io.write_string("green\n", !IO) else true ),
    id(temp(2.5), temp(T)),
    io.write_float(T, !IO),
    io.nl(!IO),
    id(node(leaf, I, node(leaf, 7, leaf)), Tree),
    show_tree(Tree, !IO),
    io.nl(!IO),
    id([1, 2], List),
    ( if List = [H | _] then io.write_int(H, !IO) else true ),
    io.nl(!IO),
    swap(1, 2.5, F2, I2),
    io.write_float(F2, !IO),
    io.write_string(" ", !IO),
    io.write_int(I2, !IO),
    io.nl(!IO),
    twice({3, "t"}, {A, B}),
    io.write_int(A, !IO),
    io.write_string(B, !IO),
    io.nl(!IO),
    io.write_int(first(9, "ignored"), !IO),
    io.nl(!IO).

% OUTPUT:42
% OUTPUT-NEXT:-1.5
% OUTPUT-NEXT:x
% OUTPUT-NEXT:-3
% OUTPUT-NEXT:200
% OUTPUT-NEXT:hello
% OUTPUT-NEXT:green
% OUTPUT-NEXT:2.5
% OUTPUT-NEXT:node(leaf, 42, node(leaf, 7, leaf))
% OUTPUT-NEXT:1
% OUTPUT-NEXT:2.5 1
% OUTPUT-NEXT:3t
% OUTPUT-NEXT:9
