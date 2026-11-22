% The collector and open terms (design document, "Open terms as built"),
% with a collection at every allocation. Open terms are traced in their
% open layout: boxes, closed(T)'s boxes and a type whose open layout shares
% another tag than its ordinary one. Unbound variable cells are copied; a
% reference to a bound one is replaced by its binding unless the binding is
% on the trail. The trail's entries follow their cells, so that bindings
% undone after a collection are undone in the cells the program sees, and
% those of cells nothing reaches any more go to a scratch word. mmc has no
% open types, so the outputs are checked by hand: each line says what it
% shows.
%
% RUN: merkc -fopen-types %s -o %t && env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && env MERK_GC_EVERY_ALLOC=1 %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module open_gc.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open.

    % Eight constructors with arguments: ordinary tags 0 to 6 and a shared
    % 7; open tags 0 to 5 and a shared 6 for the last three.
:- type eight ---> p0(int) ; p1(int) ; p2(int) ; p3(int) ; p4(int)
    ; p5(int) ; p6(int) ; p7(int).
:- type item ---> item(string, float, closed(list(int)), eight).

main(!IO) :-
    layouts(!IO), chains(!IO), unbound(!IO), undone(!IO), dead(!IO),
    poly(!IO).

:- pred list_eq(open(list(int))::ia, open(list(int))::ia) is semidet.
:- pragma no_inline(list_eq/2).
list_eq(X, Y) :- X = Y.

:- pred items_eq(open(list(item))::ia, open(list(item))::ia) is semidet.
:- pragma no_inline(items_eq/2).
items_eq(X, Y) :- X = Y.

    % I to N, or holes where I mod K = 0.
:- pred build(int::in, int::in, int::in, open(list(int))::ia) is semidet.
build(I, N, K, L) :-
    ( if I > N then
        L = []
    else if I mod K = 0 then
        L = [_ | T],
        build(I + 1, N, K, T)
    else
        L = [I | T],
        build(I + 1, N, K, T)
    ).

    % Allocates a ground list of N cells, which collects N times.
:- func churn(int) = int.
:- pragma no_inline(churn/1).
churn(N) = list.length(list.duplicate(N, N)).

    % Every field's encoding, each item's fields holes filled by
    % unification, then collected many times before from_open reads them.
:- func items = open(list(item)).
items = [item("a", 1.5, closed([1]), p6(6)), item("b", -2.0, closed([]), p5(5)),
    item("c", 0.25, closed([2, 3]), p7(7))].

:- pred layouts(io::di, io::uo) is det.
layouts(!IO) :-
    promise_pure ( if
        H = [item(_, _, _, _), item(_, _, _, p5(_)), item("c", _, _, _)]
            : open(list(item)),
        items_eq(H, items),
        churn(100) = 100,
        impure from_open(H, G)
    then
        io.write(G, !IO), io.nl(!IO)
    else
        io.write_string("layouts: no\n", !IO)
    ).
% OUTPUT: [item("a", 1.5, closed([1]), p6(6)), item("b", -2.0, closed([]), p5(5)), item("c", 0.25, closed([2, 3]), p7(7))]

    % Holes bound to each other in a chain, the last to a value: the
    % collections shorten the chain, outside any condition.
:- pred chains(io::di, io::uo) is det.
chains(!IO) :-
    promise_pure (
        L = [A, B, C, D] : open(list(int)),
        A = B, B = C, C = D, D = 4,
        X = churn(50),
        ( if impure from_open(L, G) then
            io.write(G, !IO), io.write_string(" ", !IO), io.write_int(X, !IO),
            io.nl(!IO)
        else
            io.write_string("chains: no\n", !IO)
        )
    ).
% OUTPUT: [4, 4, 4, 4] 50

    % Unbound variables survive collections as variables.
:- pred unbound(io::di, io::uo) is det.
unbound(!IO) :-
    promise_pure (
        L = [_, 2, _] : open(list(int)),
        X = churn(50),
        ( if impure from_open(L, _) then
            io.write_string("unbound: ground\n", !IO)
        else if list_eq(L, [1, 2, 3]), impure from_open(L, G) then
            io.write(G, !IO), io.write_string(" ", !IO), io.write_int(X, !IO),
            io.nl(!IO)
        else
            io.write_string("unbound: no\n", !IO)
        )
    ).
% OUTPUT: [1, 2, 3] 50

    % Bindings on the trail across collections: a condition binds A, the
    % collections move A's cell and the trail's entry with it, and the
    % condition then fails, which undoes the binding in the moved cell.
:- pred undone(io::di, io::uo) is det.
undone(!IO) :-
    promise_pure (
        L = [A, 2] : open(list(int)),
        ( if list_eq(L, [1, 2]), churn(50) = 0 then
            io.write_string("undone: churn\n", !IO)
        else if impure var(A) then
            io.write_string("undone: A unbound\n", !IO)
        else
            io.write_string("undone: A bound\n", !IO)
        ),
        ( if list_eq(L, [7, 2]), impure from_open(L, G) then
            io.write(G, !IO), io.nl(!IO)
        else
            io.write_string("undone: no\n", !IO)
        )
    ).
% OUTPUT: undone: A unbound
% OUTPUT-NEXT: [7, 2]

    % Bindings on the trail of cells nothing reaches when the collections
    % run: their entries go to a scratch word, which undoing writes.
:- pred dead(io::di, io::uo) is det.
dead(!IO) :-
    promise_pure ( if
        build(1, 100, 1, Holes), build(1, 100, 101, Full),
        list_eq(Holes, Full), churn(50) = 0
    then
        io.write_string("dead: churn\n", !IO)
    else
        io.write_string("dead: undone\n", !IO)
    ).
% OUTPUT: dead: undone

    % Polymorphic roots: open(list(T))'s type_info is a kept one, made by
    % the runtime without a safepoint, as is open(T)'s when T's type_info,
    % list(U)'s here, is a cell its caller made on the heap, which the
    % collections move.
:- pred grow(T::in, open(list(T))::ia, int::in) is semidet.
:- pragma no_inline(grow/3).
grow(X, L, N) :-
    ( if N = 0 then
        L = []
    else
        L = [to_open(X) | T],
        grow(X, T, N - 1)
    ).

:- pred wrapped(U::in, list(list(U))::out) is semidet.
:- pragma no_inline(wrapped/2).
wrapped(X, G) :-
    promise_pure ( grow([X], L, 3), impure from_open(L, G) ).

:- pred poly(io::di, io::uo) is det.
poly(!IO) :-
    promise_pure ( if grow(7, L, 3), impure from_open(L, G) then
        io.write(G : list(int), !IO), io.nl(!IO)
    else
        io.write_string("poly: no\n", !IO)
    ),
    ( if wrapped("w", W) then
        io.write(W, !IO), io.nl(!IO)
    else
        io.write_string("wrapped: no\n", !IO)
    ).
% OUTPUT: [7, 7, 7]
% OUTPUT-NEXT: {{\[\[}}"w"], ["w"], ["w"]]
