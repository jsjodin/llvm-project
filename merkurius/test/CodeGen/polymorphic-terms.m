% Terms with type variables: one layout for every instance of a type, a field
% of a type variable being a word, so that list(int) is passed as list(T),
% and a type_info for list(T) is built at run time. The output is mmc's for
% the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module polymorphic_terms.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int8, list.

:- type colour ---> red ; green ; blue.
:- type box(T) ---> box(T).
:- type tree(K, V) ---> leaf ; node(tree(K, V), K, V, tree(K, V)).
:- type many(T)
    ---> k0 ; k1 ; a(T) ; b(T, T) ; c(T) ; d(T) ; e(T) ; f(T) ; g(T) ; h(T, int) ; i(T).

% MERK-DAG: merk.type_def "list.list(T0)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(!merk.word, !merk.term<"list.list(T0)">)>]
% MERK-DAG: merk.proc private @polymorphic_terms.len_f1_0(%arg0: !merk.type_info, %arg1: !merk.term<"list.list(T0)">) -> i64

:- func len(list(T)) = int.
len([]) = 0.
len([_ | T]) = 1 + len(T).

:- pred app(list(T)::in, list(T)::in, list(T)::out) is det.
app([], L, L).
app([H | T], L, [H | R]) :-
    app(T, L, R).

:- func rev(list(T)) = list(T).
rev(L) = rev_acc(L, []).

:- func rev_acc(list(T), list(T)) = list(T).
rev_acc([], A) = A.
rev_acc([H | T], A) = rev_acc(T, [H | A]).

:- func head_or(list(T), T) = T.
head_or([], D) = D.
head_or([H | _], _) = H.

% A list of lists of T: its type_info is a cell, of a cell, of a parameter.
% MERK-LABEL: merk.proc private @polymorphic_terms.lens_f1_0
% MERK: %[[L:.*]] = merk.type_info "list.list/1"(%arg0)
% MERK: merk.call @polymorphic_terms.len_f1_0(%[[L]],
:- func lens(list(T)) = int.
lens(L) = len([L, L, rev(L)]).

:- func unbox(box(T)) = T.
unbox(box(X)) = X.

:- func rebox(box(box(T))) = box(T).
rebox(box(B)) = B.

:- func size(tree(K, V)) = int.
size(leaf) = 0.
size(node(L, _, _, R)) = size(L) + 1 + size(R).

:- func root(tree(K, V), V) = V.
root(leaf, D) = D.
root(node(_, _, V, _), _) = V.

:- func flip(tree(K, V)) = tree(V, K).
flip(leaf) = leaf.
flip(node(L, K, V, R)) = node(flip(L), V, K, flip(R)).

% Every primary tag, and the shared one's secondary tags.
:- func inner(many(T), T) = T.
inner(k0, D) = D.
inner(k1, D) = D.
inner(a(X), _) = X.
inner(b(_, X), _) = X.
inner(c(X), _) = X.
inner(d(X), _) = X.
inner(e(X), _) = X.
inner(f(X), _) = X.
inner(g(X), _) = X.
inner(h(X, _), _) = X.
inner(i(X), _) = X.

:- func tuple_second({A, B}) = B.
tuple_second({_, B}) = B.

:- func first_elem(array(T)) = T.
:- mode first_elem(array_ui) = out is det.
first_elem(A) = array.lookup(A, 0).

:- func set_first(array(T), T) = array(T).
:- mode set_first(array_di, in) = array_uo is det.
set_first(A0, X) = A :-
    array.set(0, X, A0, A).

:- func filled(int, T) = array(T).
:- mode filled(in, in) = array_uo is det.
filled(N, X) = A :-
    array.init(N, X, A).

:- pred write_ints(list(int)::in, io::di, io::uo) is det.
write_ints([], !IO) :-
    io.nl(!IO).
write_ints([H | T], !IO) :-
    io.write_int(H, !IO),
    io.write_string(" ", !IO),
    write_ints(T, !IO).

:- func iota(int, int) = list(int).
iota(I, N) = ( if I >= N then [] else [I | iota(I + 1, N)] ).

main(!IO) :-
    app([1, 2], [3], L1),
    write_ints(rev(L1), !IO),
    write_ints(rev(iota(0, 40)), !IO),
    io.write_int(len([1.5, 2.5]) + len(["a"]) + len([red, blue]), !IO),
    io.nl(!IO),
    io.write_int(len([]) + lens([1, 2, 3]), !IO),
    io.nl(!IO),
    % Each kind of value through list(T), and out again as itself.
    io.write_float(head_or(rev([1.5, -2.5]), 0.0), !IO),
    io.nl(!IO),
    io.write_int8(head_or([-3i8, 4i8], 0i8), !IO),
    io.nl(!IO),
    io.write_char(head_or(rev(['p', 'q']), 'z'), !IO),
    io.nl(!IO),
    io.write_string(head_or(["s"], "d"), !IO),
    io.nl(!IO),
    ( if head_or([green], red) = green then
        io.write_string("green\n", !IO)
    else
        io.write_string("not green\n", !IO)
    ),
    write_ints(head_or(rev([[1], [2, 3]]), []), !IO),
    % A notag type: at float its LLVM type is a float, at T a word.
    io.write_float(unbox(box(6.25)), !IO),
    io.nl(!IO),
    io.write_float(unbox(rebox(box(box(-0.5)))), !IO),
    io.nl(!IO),
    io.write_int8(unbox(box(-7i8)), !IO),
    io.nl(!IO),
    Tree = node(node(leaf, 1, 1.5, leaf), 2, 2.5, leaf),
    io.write_int(size(Tree), !IO),
    io.nl(!IO),
    io.write_float(root(Tree, 0.0), !IO),
    io.nl(!IO),
    io.write_int(root(flip(Tree), 0), !IO),
    io.nl(!IO),
    io.write_int(inner(a(1), 0) + inner(b(0, 2), 0) + inner(c(3), 0)
        + inner(d(4), 0) + inner(e(5), 0) + inner(f(6), 0) + inner(g(7), 0)
        + inner(h(8, 0), 0) + inner(i(9), 0) + inner(k1, 10), !IO),
    io.nl(!IO),
    io.write_string(tuple_second({1, "tuple"}), !IO),
    io.nl(!IO),
    array.init(3, 1.25, FA0),
    io.write_float(first_elem(FA0), !IO),
    io.nl(!IO),
    FA = set_first(FA0, 9.5),
    io.write_float(array.lookup(FA, 0), !IO),
    io.nl(!IO),
    NA = filled(2, -9i8),
    io.write_int8(array.lookup(NA, 1), !IO),
    io.nl(!IO).

% OUTPUT:3 2 1 
% OUTPUT-NEXT:39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 
% OUTPUT-NEXT:5
% OUTPUT-NEXT:3
% OUTPUT-NEXT:-2.5
% OUTPUT-NEXT:-3
% OUTPUT-NEXT:q
% OUTPUT-NEXT:s
% OUTPUT-NEXT:green
% OUTPUT-NEXT:2 3 
% OUTPUT-NEXT:6.25
% OUTPUT-NEXT:-0.5
% OUTPUT-NEXT:-7
% OUTPUT-NEXT:2
% OUTPUT-NEXT:2.5
% OUTPUT-NEXT:2
% OUTPUT-NEXT:55
% OUTPUT-NEXT:tuple
% OUTPUT-NEXT:1.25
% OUTPUT-NEXT:9.5
% OUTPUT-NEXT:-9

% A procedure nothing calls has its own type_def for list(T).
:- pred no_calls(list(T)::in) is semidet.
no_calls([]).
