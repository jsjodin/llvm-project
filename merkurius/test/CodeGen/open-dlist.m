% Difference lists with open terms (design document, "Open terms as built"),
% run: the design document's two examples ("Open terms", Example and Example:
% appending difference lists), a tree flattened in order through a hole passed
% down, and a queue that adds at its end through its hole. Closing a hole twice
% fails the second time. mmc rejects building a difference list (design
% document, E4), so the outputs are checked by hand: each line says what it
% shows.
%
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module open_dlist.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, pair.

main(!IO) :-
    first_example(!IO), second_example(!IO), flatten(!IO), queue(!IO).

%---------------------------------------------------------------------------%
% The design document's first example, and a second close of the same hole.

:- pred make_open(list(int)::in,
    open(list(int))::oa, open(list(int))::oa) is det.
make_open([], T, T).
make_open([X | Xs], [X | R], T) :-
    make_open(Xs, R, T).

    % Fill the hole with Tail and return the ground list.
:- pred close(open(list(int))::ia, open(list(int))::ia,
    list(int)::in, list(int)::out) is semidet.
:- pragma promise_pure(close/4).
close(Front, Hole, Tail, List) :-
    Hole = to_open(Tail),
    impure from_open(Front, List).

:- pred first_example(io::di, io::uo) is det.
first_example(!IO) :-
    make_open([1, 2, 3], Front, Hole),
    promise_pure (
        ( if close(Front, Hole, [4, 5], List) then
            io.write_line(List, !IO)
        else
            io.write_string("hole already bound\n", !IO)
        ),
        ( if close(Front, Hole, [6], List2) then
            io.write_line(List2, !IO)
        else
            io.write_string("hole already bound\n", !IO)
        )
    ).
% OUTPUT: [1, 2, 3, 4, 5]
% OUTPUT-NEXT: hole already bound

%---------------------------------------------------------------------------%
% The design document's second example: app(A-B, B-C, A-C).

:- type dlist == open(pair(list(int), list(int))).   % Front - Hole

:- pred append_dl(dlist::ia, dlist::ia, dlist::oa) is semidet.
append_dl(A - B, B - C, A - C).

    % Close the hole with [] and return an ordinary list.
:- pred dl_to_list(dlist::ia, list(int)::out) is semidet.
:- pragma promise_pure(dl_to_list/2).
dl_to_list(F - [], L) :-
    impure from_open(F, L).

:- pred test(list(int)::out) is semidet.
test(L) :-
    D1 = [1, 2 | H1] - H1,
    D2 = [3 | H2] - H2,
    append_dl(D1, D2, D),
    dl_to_list(D, L).

:- pred first(int::out) is semidet.
first(X) :-
    test(L),
    L = [X | _].

:- pred second_example(io::di, io::uo) is det.
second_example(!IO) :-
    ( if test(L), first(X) then
        io.write(L, !IO), io.write_string(" ", !IO), io.write_line(X, !IO)
    else
        io.write_string("second example: no\n", !IO)
    ).
% OUTPUT-NEXT: [1, 2, 3] 1

%---------------------------------------------------------------------------%
% A tree flattened in order: each node puts its element in front of the
% right subtree's list, whose hole is the one passed down, and the left
% subtree's list ends in that.

:- type tree ---> leaf ; node(tree, int, tree).

:- pred flat(tree::in, open(list(int))::oa, open(list(int))::ia) is det.
flat(leaf, H, H).
flat(node(L, X, R), F, H) :-
    flat(R, H2, H),
    flat(L, F, [X | H2]).

:- func build(int, int) = tree.
build(Lo, Hi) = T :-
    ( if Lo > Hi then
        T = leaf
    else
        Mid = (Lo + Hi) / 2,
        T = node(build(Lo, Mid - 1), Mid, build(Mid + 1, Hi))
    ).

:- pred flatten(io::di, io::uo) is det.
flatten(!IO) :-
    promise_pure (
        Hole = _ : open(list(int)),
        flat(build(1, 9), Front, Hole),
        ( if Hole = [10, 11], impure from_open(Front, List) then
            io.write_line(List, !IO)
        else
            io.write_string("flatten: no\n", !IO)
        ),
        flat(build(1, 2000), Front2, []),
        ( if impure from_open(Front2, List2) then
            io.write_line(list.foldl(plus, List2, 0), !IO)
        else
            io.write_string("flatten: no\n", !IO)
        )
    ).
% OUTPUT-NEXT: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
% OUTPUT-NEXT: 2001000

%---------------------------------------------------------------------------%
% A queue: its front, and the hole at its end, where put binds a new cell.

:- type queue == open(pair(list(int), list(int))).

:- pred empty(queue::oa) is det.
empty(H - H).

:- pred put(int::in, queue::ia, queue::oa) is semidet.
put(X, F - [X | H], F - H).

:- pred puts(list(int)::in, queue::ia, queue::oa) is semidet.
puts([], Q, Q).
puts([X | Xs], Q0, Q) :-
    put(X, Q0, Q1),
    puts(Xs, Q1, Q).

:- pred queue(io::di, io::uo) is det.
queue(!IO) :-
    promise_pure (
        empty(Q0),
        ( if
            puts([1, 2, 3], Q0, Q1),
            puts(1 .. 1000, Q1, Q2),
            dl_to_list(Q2, L)
        then
            io.write_line(list.take_upto(5, L) : list(int), !IO),
            io.write_line(list.length(L) : int, !IO)
        else
            io.write_string("queue: no\n", !IO)
        )
    ).
% OUTPUT-NEXT: [1, 2, 3, 1, 2]
% OUTPUT-NEXT: 1003
