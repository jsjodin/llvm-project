% Values of types defined by constructors (design document, "Type
% representation"): merk.type_def lists each type's constructors, and
% lowering chooses the layout. A type of constants only is a word, as
% colour; one of one constructor of one argument is the argument, as
% temp and celsius, which are a float; any other is a tagged word,
% constants shifted past the 3-bit primary tag, and a cell of each other
% constructor's arguments, with no header word. many has two constants
% and nine constructors with arguments, so tags 1 to 6 are a(_) to f(_)
% and tag 7 is shared by g(_), h(_, _) and i(_), whose cells' first words
% are their secondary tags 0, 1 and 2. A unification of two terms calls
% the type's unify procedure. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK: merk.type_def "terms.many" [#merk.constructor<"k0">, #merk.constructor<"a"(i64)>, #merk.constructor<"b"(i64, i64)>, #merk.constructor<"c"(f64)>, #merk.constructor<"d"(!merk.term<"terms.many">)>, {{.*}}, #merk.constructor<"k1">]
% MERK: merk.type_def "list.list(int)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]
% MERK: merk.type_def "terms.tree" [#merk.constructor<"leaf">, #merk.constructor<"node"(!merk.term<"terms.tree">, i64, !merk.term<"terms.tree">)>]
% MERK: merk.type_def "{int, int}" [#merk.constructor<"{}"(i64, i64)>]
% MERK: merk.type_def "terms.temp" [#merk.constructor<"temp"(!merk.term<"terms.celsius">)>]
% MERK-LABEL: merk.proc private @terms.insert_f2_0(
% MERK: merk.construct #merk.functor<"leaf", 0> -> !merk.term<"terms.tree">
% MERK: merk.construct #merk.functor<"node", 3>(%{{.*}}, %arg0, %{{.*}} : !merk.term<"terms.tree">, i64, !merk.term<"terms.tree">) -> !merk.term<"terms.tree">
% MERK-LABEL: merk.proc private @terms.is_a_2_0(%arg0: !merk.term<"terms.many">) -> i64 attributes {determinism = #merk.determinism<semidet>} {
% MERK-NEXT: %[[IS:.*]] = merk.has_functor #merk.functor<"a", 1> %arg0 : !merk.term<"terms.many">
% MERK-NEXT: merk.guard %[[IS]]
% MERK-NEXT: %[[X:.*]] = merk.deconstruct #merk.functor<"a", 1> %arg0 : !merk.term<"terms.many"> -> i64
% MERK-NEXT: merk.succeed %[[X]] : i64
% MERK-LABEL: merk.proc private @terms.value_f1_0(%arg0: !merk.term<"terms.many">) -> i64
% MERK-NEXT: merk.switch %arg0 : !merk.term<"terms.many"> -> i64 cases {{\[}}[#merk.functor<"a", 1>], [#merk.functor<"b", 2>],
% MERK-NEXT: merk.deconstruct #merk.functor<"a", 1> %arg0 : !merk.term<"terms.many"> -> i64
% MERK-LABEL: merk.proc private @terms.class_f1_0(
% MERK-NEXT: merk.switch %arg0 : !merk.term<"terms.many"> -> i64 cases {{\[}}[#merk.functor<"a", 1>, #merk.functor<"e", 1>, #merk.functor<"i", 1>],
% MERK-LABEL: merk.proc private @terms.same_2_0(
% MERK-NEXT: merk.call @unify.terms.many(%arg1, %arg0) : (!merk.term<"terms.many">, !merk.term<"terms.many">) -> ()
% MERK-LABEL: merk.proc private @unify.terms.many(%arg0: !merk.term<"terms.many">, %arg1: !merk.term<"terms.many">) attributes {determinism = #merk.determinism<semidet>} {
% MERK-NEXT: merk.switch %arg0 : !merk.term<"terms.many"> cases {{\[}}[#merk.functor<"k0", 0>], [#merk.functor<"a", 1>],
% MERK-NEXT: %[[K0:.*]] = merk.has_functor #merk.functor<"k0", 0> %arg1
% MERK-NEXT: merk.guard %[[K0]]
% MERK-NEXT: merk.yield
% MERK-NEXT: }, {
% MERK-NEXT: %[[A:.*]] = merk.has_functor #merk.functor<"a", 1> %arg1
% MERK-NEXT: merk.guard %[[A]]
% MERK-NEXT: %[[XA:.*]] = merk.deconstruct #merk.functor<"a", 1> %arg0
% MERK-NEXT: %[[YA:.*]] = merk.deconstruct #merk.functor<"a", 1> %arg1
% MERK-NEXT: %[[EQ:.*]] = arith.cmpi eq, %[[XA]], %[[YA]] : i64
% MERK-NEXT: merk.guard %[[EQ]]
% MERK: merk.call @unify.terms.many(%{{.*}}, %{{.*}}) : (!merk.term<"terms.many">, !merk.term<"terms.many">) -> ()

% LLVM: @merk_no_secondary_tag = internal constant i64 -1
% A constant's word is its number shifted past the tag, and a cell is
% tagged by its constructor's tag.
% LLVM-LABEL: define internal tailcc ptr @terms.insert_f2_0(i64 %0, ptr %1) {
% LLVM: %[[CELL:.*]] = call ptr @merk_alloc(i64 24)
% LLVM-NEXT: store ptr null, ptr %[[CELL]], align 8
% LLVM-NEXT: %[[V:.*]] = getelementptr i8, ptr %[[CELL]], i32 8
% LLVM-NEXT: store i64 %0, ptr %[[V]], align 8
% LLVM-NEXT: %[[R:.*]] = getelementptr i8, ptr %[[CELL]], i32 16
% LLVM-NEXT: store ptr null, ptr %[[R]], align 8
% LLVM-NEXT: %[[NODE:.*]] = getelementptr i8, ptr %[[CELL]], i32 1
% LLVM-NEXT: ret ptr %[[NODE]]
% Constants only: a word.
% LLVM-LABEL: define internal tailcc i64 @terms.next_f1_0(i64 %0) {
% LLVM-NEXT: switch i64 %0, label %{{.*}} [
% A tuple is a cell with tag 0.
% LLVM-LABEL: define internal tailcc ptr @terms.swap_f1_0(ptr %0) {
% LLVM-NEXT: %[[X:.*]] = load i64, ptr %0, align 8
% LLVM-NEXT: %[[YP:.*]] = getelementptr i8, ptr %0, i32 8
% LLVM-NEXT: %[[Y:.*]] = load i64, ptr %[[YP]], align 8
% LLVM-NEXT: %[[T:.*]] = call ptr @merk_alloc(i64 16)
% LLVM-NEXT: store i64 %[[Y]], ptr %[[T]], align 8
% LLVM: ret ptr %[[T]]
% One constructor of one argument: the argument.
% LLVM-LABEL: define internal tailcc double @terms.warm_f1_0(double %0) {
% LLVM-NEXT: fadd double %0, 1.000000e+00
% A switch on the primary tag, then on the constants' words.
% LLVM-LABEL: define internal tailcc i64 @terms.value_f1_0(ptr %0) {
% LLVM-NEXT: %[[W:.*]] = ptrtoint ptr %0 to i64
% LLVM-NEXT: %[[TAG:.*]] = and i64 %[[W]], 7
% LLVM-NEXT: %[[WORD:.*]] = ptrtoint ptr %0 to i64
% LLVM-NEXT: switch i64 %[[TAG]], label %{{.*}} [
% LLVM-NEXT: i64 1, label %{{.*}}
% LLVM: i64 6, label %{{.*}}
% LLVM-NEXT: i64 0, label %[[CONSTANTS:.*]]
% LLVM-NEXT: i64 7, label %[[SHARED:.*]]
% LLVM-NEXT: ]
% A field is at its word less the tag.
% LLVM-EMPTY:
% LLVM-NEXT: {{.*}}:
% LLVM-NEXT: getelementptr i8, ptr %0, i32 -1
% LLVM: [[CONSTANTS]]:
% LLVM-NEXT: switch i64 %[[WORD]], label %{{.*}} [
% LLVM-NEXT: i64 0, label %{{.*}}
% LLVM-NEXT: i64 8, label %{{.*}}
% LLVM: [[SHARED]]:
% LLVM-NEXT: %[[SP:.*]] = getelementptr i8, ptr %0, i32 -7
% LLVM-NEXT: %[[S:.*]] = load i64, ptr %[[SP]], align 8
% LLVM-NEXT: switch i64 %[[S]], label %{{.*}} [
% LLVM-NEXT: i64 0, label %{{.*}}
% LLVM-NEXT: i64 1, label %{{.*}}
% LLVM-NEXT: i64 2, label %{{.*}}
% A test of a shared tag reads the secondary tag, or a word that is none
% if the term is a constant, which has no cell.
% LLVM-LABEL: define internal tailcc i1 @terms.is_g_1_0(ptr %0) {
% LLVM: %[[IS7:.*]] = icmp eq i64 %{{.*}}, 7
% LLVM-NEXT: %[[P:.*]] = getelementptr i8, ptr %0, i32 -7
% LLVM-NEXT: %[[AT:.*]] = select i1 %[[IS7]], ptr %[[P]], ptr @merk_no_secondary_tag
% LLVM-NEXT: %[[SEC:.*]] = load i64, ptr %[[AT]], align 8
% LLVM-NEXT: %[[IS0:.*]] = icmp eq i64 %[[SEC]], 0
% LLVM-NEXT: and i1 %[[IS7]], %[[IS0]]
% LLVM: declare ptr @merk_alloc(i64)

% OUTPUT:100 0
% OUTPUT-NEXT:1 1
% OUTPUT-NEXT:3 3
% OUTPUT-NEXT:3 3
% OUTPUT-NEXT:8 3
% OUTPUT-NEXT:6 1
% OUTPUT-NEXT:8 3
% OUTPUT-NEXT:10 3
% OUTPUT-NEXT:109 2
% OUTPUT-NEXT:90 1
% OUTPUT-NEXT:101 2
% OUTPUT-NEXT:108 2
% OUTPUT-NEXT:10 3
% OUTPUT-NEXT:6 3
% OUTPUT-NEXT:2 3
% OUTPUT-NEXT:=..............
% OUTPUT-NEXT:.=.............
% OUTPUT-NEXT:..=............
% OUTPUT-NEXT:...=...........
% OUTPUT-NEXT:....=..........
% OUTPUT-NEXT:.....=.........
% OUTPUT-NEXT:......=........
% OUTPUT-NEXT:.......=.......
% OUTPUT-NEXT:........=......
% OUTPUT-NEXT:.........=.....
% OUTPUT-NEXT:..........=....
% OUTPUT-NEXT:...........=...
% OUTPUT-NEXT:............=..
% OUTPUT-NEXT:.............=.
% OUTPUT-NEXT:..............=
% OUTPUT-NEXT:45
% OUTPUT-NEXT:4
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:42
% OUTPUT-NEXT:-1
% OUTPUT-NEXT:-a-----g-------
% OUTPUT-NEXT:3
% OUTPUT-NEXT:21
% OUTPUT-NEXT:21.5
% OUTPUT-NOT:{{.}}

:- module terms.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, float, list.

% Nine constructors with arguments and two constants: tags 1 to 6 alone,
% 7 shared by three, with secondary tags.
:- type many
    --->    k0
    ;       a(int)
    ;       b(int, int)
    ;       c(float)
    ;       d(many)
    ;       e(int)
    ;       f(int)
    ;       g(int)
    ;       h(int, many)
    ;       i(int)
    ;       k1.

:- type tree ---> leaf ; node(tree, int, tree).
:- type celsius ---> celsius(float).
:- type temp ---> temp(celsius).
:- type colour ---> red ; green ; blue.
:- type point == {int, int}.

:- func value(many) = int.
value(k0) = 100.
value(k1) = 101.
value(a(X)) = X.
value(b(X, Y)) = X - Y.
value(c(F)) = truncate_to_int(F).
value(d(M)) = 2 * value(M).
value(e(X)) = X + 1.
value(f(X)) = X + 2.
value(g(X)) = X + 3.
value(h(X, M)) = X + value(M).
value(i(X)) = X * 10.

:- func class(many) = int.
class(M) = R :-
    ( M = k0, R = 0
    ; ( M = a(_) ; M = e(_) ; M = i(_) ), R = 1
    ; ( M = k1 ; M = h(_, _) ), R = 2
    ; ( M = b(_, _) ; M = c(_) ; M = d(_) ; M = f(_) ; M = g(_) ), R = 3
    ).

:- func insert(int, tree) = tree.
insert(X, leaf) = node(leaf, X, leaf).
insert(X, node(L, V, R)) =
    ( if X < V then node(insert(X, L), V, R)
    else if X > V then node(L, V, insert(X, R))
    else node(L, V, R) ).

:- func total(tree) = int.
total(leaf) = 0.
total(node(L, V, R)) = total(L) + V + total(R).

:- func depth(tree) = int.
depth(leaf) = 0.
depth(node(L, _, R)) = 1 + max(depth(L), depth(R)).

:- pred is_a(many::in, int::out) is semidet.
is_a(a(X), X).

:- pred is_g(many::in) is semidet.
is_g(g(_)).

:- func next(colour) = colour.
next(red) = green.
next(green) = blue.
next(blue) = red.

:- func colour_code(colour) = int.
colour_code(C) = ( if C = red then 1 else if C = green then 2 else 3 ).

:- func swap(point) = point.
swap({X, Y}) = {Y, X}.

:- func warm(temp) = float.
warm(temp(celsius(C))) = C + 1.0.

:- pred show(int::in, io::di, io::uo) is det.
show(X, !IO) :- io.write_int(X, !IO), io.nl(!IO).


:- pred yes_no(bool_test::in, io::di, io::uo) is det.
:- type bool_test ---> yes ; no.
yes_no(yes, !IO) :- io.write_string("yes\n", !IO).
yes_no(no, !IO) :- io.write_string("no\n", !IO).

:- pred same(many::in, many::in) is semidet.
same(X, X).

:- pred same_tree(tree::in, tree::in) is semidet.
same_tree(X, X).

:- func inserts(list(int), tree) = tree.
inserts([], T) = T.
inserts([X | Xs], T) = inserts(Xs, insert(X, T)).

:- pred unify_all(list(many)::in, list(many)::in, io::di, io::uo) is det.
unify_all([], _, !IO).
unify_all([X | Xs], Ys, !IO) :-
    unify_with(X, Ys, !IO), io.nl(!IO),
    unify_all(Xs, Ys, !IO).

:- pred unify_with(many::in, list(many)::in, io::di, io::uo) is det.
unify_with(_, [], !IO).
unify_with(X, [Y | Ys], !IO) :-
    io.write_string(( if same(X, Y) then "=" else "." ), !IO),
    unify_with(X, Ys, !IO).

main(!IO) :-
    Ms = [k0, a(1), b(5, 2), c(3.7), d(a(4)), e(5), f(6), g(7),
        h(8, k1), i(9), k1, h(8, k0), d(a(5)), d(c(3.7)), b(5, 3)],
    show_all(Ms, !IO),
    unify_all(Ms, Ms, !IO),
    T = inserts([5, 3, 8, 1, 4, 7, 9, 2, 6], leaf),
    show(total(T), !IO),
    show(depth(T), !IO),
    yes_no(( if same_tree(T, insert(4, T)) then yes else no ), !IO),
    yes_no(( if same_tree(T, insert(10, T)) then yes else no ), !IO),
    ( if is_a(a(42), A) then show(A, !IO) else show(-1, !IO) ),
    ( if is_a(b(1, 2), B) then show(B, !IO) else show(-1, !IO) ),
    show_g(Ms, !IO),
    show(colour_code(next(next(red))), !IO),
    {P, Q} = swap({1, 2}),
    show(P * 10 + Q, !IO),
    io.write_float(warm(temp(celsius(20.5))), !IO), io.nl(!IO).

:- pred show_g(list(many)::in, io::di, io::uo) is det.
show_g([], !IO) :- io.nl(!IO).
show_g([M | Ms], !IO) :-
    io.write_string(( if is_g(M) then "g" else if is_a(M, _) then "a" else "-" ), !IO),
    show_g(Ms, !IO).

:- pred show_all(list(many)::in, io::di, io::uo) is det.
show_all([], !IO).
show_all([M | Ms], !IO) :-
    io.write_int(value(M), !IO), io.write_char(' ', !IO),
    io.write_int(class(M), !IO), io.nl(!IO),
    show_all(Ms, !IO).
