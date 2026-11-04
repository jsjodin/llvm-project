% Unifications of values of a type variable, by way of the type_info's
% descriptor (design document, "Type descriptors"): each kind of value at T,
% terms holding them, a type_info built at run time, and arrays, strings and
% tuples, at known types too. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module polymorphic_unify.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int8, list, uint8.

:- type colour ---> red ; green ; blue.
:- type temp ---> temp(float).
:- type tree ---> leaf ; node(tree, int, tree).
:- type box(T) ---> box(T).
:- type pt(A, B) ---> pt(A, B) ; none.

% MERK-DAG: merk.type_ctor "int/0" module "" name "int" kind "int" linkage "linkonce" arity 0 unify @"unify_entry.int/0"
% MERK-DAG: merk.type_ctor "list.list/1" module "list" name "list" kind "term" linkage "linkonce" arity 1 term !merk.term<"list.list(T0)"> unify @"unify_entry.list.list/1"
% MERK-DAG: merk.type_ctor "polymorphic_unify.tree/0" module "polymorphic_unify" name "tree" kind "term" linkage "internal" arity 0 term !merk.term<"polymorphic_unify.tree"> unify @"unify_entry.polymorphic_unify.tree/0"
% LLVM-DAG: @"merk.type_ctor.int/0" = linkonce_odr constant { i8, i8, i16, i32, ptr, ptr, ptr, ptr } { i8 0, i8 0, i16 0, i32 0, ptr @{{.*}}, ptr @{{.*}}, ptr @"unify_entry.int/0", ptr @"compare_entry.int/0" }
% LLVM-DAG: @"merk.type_ctor.float/0" = linkonce_odr constant { i8, i8, i16, i32, ptr, ptr, ptr, ptr } { i8 10,
% LLVM-DAG: @"merk.type_ctor.polymorphic_unify.colour/0" = internal constant { i8, i8, i16, i32, ptr, ptr, ptr, ptr } { i8 16,
% LLVM-DAG: @"merk.type_ctor.polymorphic_unify.temp/0" = internal constant { i8, i8, i16, i32, ptr, ptr, ptr, ptr } { i8 17,
% LLVM-DAG: @"merk.type_ctor.polymorphic_unify.tree/0" = internal constant { i8, i8, i16, i32, ptr, ptr, ptr, ptr } { i8 18,

% MERK-DAG: merk.proc private @polymorphic_unify.same_2_0(%arg0: !merk.type_info, %arg1: !merk.word, %arg2: !merk.word)
% MERK-DAG: merk.unify_word %arg0, %arg1, %arg2
:- pred same(T::in, T::in) is semidet.
same(X, Y) :- X = Y.

% Its unify procedure takes a type_info for T first.
% MERK-DAG: merk.proc private @"unify.list.list(T0)"(%arg0: !merk.type_info, %arg1: !merk.term<"list.list(T0)">, %arg2: !merk.term<"list.list(T0)">)
:- pred same_list(list(T)::in, list(T)::in) is semidet.
same_list(X, Y) :- X = Y.

% Its type_info for list(T) is made here, at run time.
:- pred same_singleton(T::in, T::in) is semidet.
same_singleton(X, Y) :- same([X], [Y]).

:- pred same_pair(A::in, B::in, A::in, B::in) is semidet.
same_pair(A1, B1, A2, B2) :- same({A1, B1}, {A2, B2}), pt(A1, B1) = pt(A2, B2).

:- pred member_of(T::in, list(T)::in) is semidet.
member_of(X, [H | T]) :-
    ( if X = H then true else member_of(X, T) ).

:- pred yn(bool_::in, io::di, io::uo) is det.
yn(yes_, !IO) :- io.write_string("y ", !IO).
yn(no_, !IO) :- io.write_string("n ", !IO).

:- type bool_ ---> yes_ ; no_.

% A string at another address, so that only its bytes can make it equal.
:- pragma foreign_decl("C", "#include <string.h>").
:- func dup(string) = string.
:- pragma foreign_proc("C", dup(X::in) = (Y::out),
    [will_not_call_mercury, promise_pure], "Y = strdup(X);").

:- func nan(float) = float.
nan(Z) = unchecked_quotient(Z, Z).

:- func t(T, T) = bool_.
t(X, Y) = ( if same(X, Y) then yes_ else no_ ).

:- func ts(T, T) = bool_.
ts(X, Y) = ( if same_singleton(X, Y) then yes_ else no_ ).

main(!IO) :-
    yn(t(1, 1), !IO), yn(t(1, 2), !IO), yn(t(-1, -1), !IO), io.nl(!IO),
    yn(t(1.5, 1.5), !IO), yn(t(1.5, 2.5), !IO), yn(t(0.0, -0.0), !IO),
    yn(t(nan(0.0), nan(0.0)), !IO), io.nl(!IO),
    yn(t('a', 'a'), !IO), yn(t('a', 'b'), !IO), io.nl(!IO),
    yn(t(-3i8, -3i8), !IO), yn(t(-3i8, 3i8), !IO), yn(t(200u8, 200u8), !IO), io.nl(!IO),
    yn(t("s", "s"), !IO), yn(t("s", "t"), !IO),
    yn(t(dup("same"), "same"), !IO), io.nl(!IO),
    yn(t(green, green), !IO), yn(t(green, blue), !IO), io.nl(!IO),
    yn(t(temp(1.5), temp(1.5)), !IO), yn(t(temp(1.5), temp(2.5)), !IO), io.nl(!IO),
    T1 = node(leaf, 1, node(leaf, 2, leaf)),
    T2 = node(leaf, 1, node(leaf, 3, leaf)),
    yn(t(T1, T1), !IO), yn(t(T1, T2), !IO), yn(t(T1, leaf), !IO), io.nl(!IO),
    yn(t([1, 2], [1, 2]), !IO), yn(t([1, 2], [1, 3]), !IO), yn(t([1], [1, 2]), !IO), io.nl(!IO),
    yn(t(["a", "b"], ["a", "b"]), !IO), yn(t([[1.5]], [[2.5]]), !IO), io.nl(!IO),
    yn(t({1, "x"}, {1, "x"}), !IO), yn(t({1, "x"}, {1, "y"}), !IO), io.nl(!IO),
    yn(t(box(1.5), box(1.5)), !IO), yn(t(box(box('p')), box(box('q'))), !IO), io.nl(!IO),
    yn(t(pt(1, "a"), pt(1, "a")), !IO), yn(t(pt(1, "a"), none), !IO), io.nl(!IO),
    yn(ts(5, 5), !IO), yn(ts("a", "b"), !IO), yn(ts(green, green), !IO),
    yn(ts([1.5], [1.5]), !IO), io.nl(!IO),
    ( if same_pair(1, "a", 1, "a") then yn(yes_, !IO) else yn(no_, !IO) ),
    ( if same_pair(1, "a", 2, "a") then yn(yes_, !IO) else yn(no_, !IO) ),
    ( if same_list(['a'], ['a']) then yn(yes_, !IO) else yn(no_, !IO) ),
    ( if member_of("c", ["a", "b", "c"]) then yn(yes_, !IO) else yn(no_, !IO) ),
    ( if member_of(4, [1, 2, 3]) then yn(yes_, !IO) else yn(no_, !IO) ),
    io.nl(!IO),
    % Arrays and strings at known types, and arrays at T.
    array.init(3, 7, A1),
    array.init(3, 7, A2),
    array.init(2, 7, A3),
    array.init(3, 8, A4),
    yn(t(A1, A2), !IO), yn(t(A1, A3), !IO), yn(t(A1, A4), !IO), io.nl(!IO),
    ( if A1 = A2 then yn(yes_, !IO) else yn(no_, !IO) ),
    ( if A1 = A4 then yn(yes_, !IO) else yn(no_, !IO) ),
    array.init(2, "s", S1),
    array.init(2, "s", S2),
    yn(t(S1, S2), !IO),
    array.init(2, -1i8, N1),
    array.init(2, -1i8, N2),
    yn(t(N1, N2), !IO),
    ( if dup("ab") = "ab" then yn(yes_, !IO) else yn(no_, !IO) ),
    ( if "ab" = "a" then yn(yes_, !IO) else yn(no_, !IO) ),
    io.nl(!IO).

% OUTPUT:y n y 
% OUTPUT-NEXT:y n y n 
% OUTPUT-NEXT:y n 
% OUTPUT-NEXT:y n y 
% OUTPUT-NEXT:y n y 
% OUTPUT-NEXT:y n 
% OUTPUT-NEXT:y n 
% OUTPUT-NEXT:y n n 
% OUTPUT-NEXT:y n n 
% OUTPUT-NEXT:y n 
% OUTPUT-NEXT:y n 
% OUTPUT-NEXT:y n 
% OUTPUT-NEXT:y n 
% OUTPUT-NEXT:y n y y 
% OUTPUT-NEXT:y n y y n 
% OUTPUT-NEXT:y n n 
% OUTPUT-NEXT:y n y y y n 
