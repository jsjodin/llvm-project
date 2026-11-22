% Open types (design document, "Open terms"; docs/INPROGRESS.md, Open
% types): a clause that does not type-check as it is is checked again with
% the open twins of its constructors, open(T) having T's constructors with
% open arguments. Each variable that can be open is, and then the fewest
% twins are used, so that a hole is open; an argument takes a ground value
% only where its type forces it. A clause that type-checks as it is, as
% main's, is typed as before. closed(T) stays closed under open(_), and
% open(open(T)) is open(T). The examples are the design document's.
%
% RUN: merkc -fc1 -fopen-types -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error --implicit-check-not=warning %s

:- module open_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module open, list, pair.

:- pred make_open(list(int)::in,
    open(list(int))::oa, open(list(int))::oa) is det.

make_open([], T, T).
make_open([X | Xs], [X | R], T) :-
    make_open(Xs, R, T).
% CHECK:      clause at [[@LINE-2]]:1 of pred open_types.make_open/3: pred open_types.make_open(list.list(int), open.open(list.list(int)), open.open(list.list(int)))
% CHECK-NEXT:   X: int
% CHECK-NEXT:   Xs: list.list(int)
% CHECK-NEXT:   R: open.open(list.list(int))
% CHECK-NEXT:   T: open.open(list.list(int))
% CHECK-NEXT:   [[@LINE-7]]:11 [X | Xs] -> constructor list.'[|]'/2
% CHECK-NEXT:   [[@LINE-8]]:21 [X | R] -> constructor list.'[|]'/2, open, argument 1 ground

:- pred close(open(list(int))::ia, open(list(int))::ia,
    list(int)::in, list(int)::out) is semidet.

close(Front, Hole, Tail, List) :-
    Hole = to_open(Tail),
    from_open(Front, List).
% CHECK:      clause at [[@LINE-3]]:1 of pred open_types.close/4
% CHECK-NEXT:   Front: open.open(list.list(int))
% CHECK-NEXT:   Hole: open.open(list.list(int))
% CHECK-NEXT:   Tail: list.list(int)
% CHECK-NEXT:   List: list.list(int)

:- type dlist == open(pair(list(int), list(int))).

:- pred append_dl(dlist::ia, dlist::ia, dlist::oa) is semidet.
append_dl(A - B, B - C, A - C).
% CHECK:      clause at [[@LINE-1]]:1 of pred open_types.append_dl/3: pred open_types.append_dl(open.open(pair.pair(list.list(int), list.list(int))), open.open(pair.pair(list.list(int), list.list(int))), open.open(pair.pair(list.list(int), list.list(int))))
% CHECK-NEXT:   A: open.open(list.list(int))
% CHECK-NEXT:   B: open.open(list.list(int))
% CHECK-NEXT:   C: open.open(list.list(int))
% CHECK-NEXT:   [[@LINE-5]]:13 - -> constructor pair.'-'/2, open
% CHECK-NEXT:   [[@LINE-6]]:20 - -> constructor pair.'-'/2, open
% CHECK-NEXT:   [[@LINE-7]]:27 - -> constructor pair.'-'/2, open

:- pred dl_to_list(dlist::ia, list(int)::out) is semidet.
dl_to_list(F - [], L) :-
    from_open(F, L).
% CHECK:      clause at [[@LINE-2]]:1 of pred open_types.dl_to_list/2
% CHECK-NEXT:   F: open.open(list.list(int))
% CHECK-NEXT:   L: list.list(int)
% CHECK-NEXT:   [[@LINE-5]]:16 [] -> constructor list.'[]'/0, open
% CHECK-NEXT:   [[@LINE-6]]:14 - -> constructor pair.'-'/2, open

    % The holes H1 and H2 are open, through the twins of the lists' cells,
    % rather than ground in ground cells.
:- pred test(list(int)::out) is semidet.
test(L) :-
    D1 = [1, 2 | H1] - H1,
    D2 = [3 | H2] - H2,
    append_dl(D1, D2, D),
    dl_to_list(D, L).
% CHECK:      clause at [[@LINE-5]]:1 of pred open_types.test/1
% CHECK-NEXT:   L: list.list(int)
% CHECK-NEXT:   D1: open.open(pair.pair(list.list(int), list.list(int)))
% CHECK-NEXT:   H1: open.open(list.list(int))
% CHECK-NEXT:   D2: open.open(pair.pair(list.list(int), list.list(int)))
% CHECK-NEXT:   H2: open.open(list.list(int))
% CHECK-NEXT:   D: open.open(pair.pair(list.list(int), list.list(int)))
% CHECK-NEXT:   [[@LINE-11]]:10 [1, 2 | H1] -> constructor list.'[|]'/2, open, argument 1 ground
% CHECK-NEXT:   [[@LINE-12]]:10 [1, 2 | H1] -> constructor list.'[|]'/2, open, argument 1 ground
% CHECK-NEXT:   [[@LINE-13]]:22 - -> constructor pair.'-'/2, open
% CHECK-NEXT:   [[@LINE-13]]:10 [3 | H2] -> constructor list.'[|]'/2, open, argument 1 ground
% CHECK-NEXT:   [[@LINE-14]]:19 - -> constructor pair.'-'/2, open

    % A closed field of an open term holds a ground value, built with the
    % ordinary constructors.
:- type big ---> big(int, int).
:- type bundle ---> bundle(closed(big), int).
:- pred fill(open(bundle)::oa) is det.
fill(bundle(closed(big(1, 2)), _)).
% CHECK:      clause at [[@LINE-1]]:1 of pred open_types.fill/1: pred open_types.fill(open.open(open_types.bundle))
% CHECK-NEXT:   [[@LINE-2]]:20 big -> constructor open_types.big/2
% CHECK-NEXT:   [[@LINE-3]]:13 closed -> constructor open.closed/1
% CHECK-NEXT:   [[@LINE-4]]:6 bundle -> constructor open_types.bundle/2, open

    % open(open(T)) is open(T), and open(closed(T)) is closed(T).
:- pred normal(open(open(int))::ia, open(closed(int))::ia) is det.
normal(_, _).
% CHECK:      clause at [[@LINE-1]]:1 of pred open_types.normal/2: pred open_types.normal(open.open(int), open.closed(int))

main(!IO) :-
    X = [1, 2],
    io.write(X, !IO).
% CHECK:      clause at [[@LINE-3]]:1 of pred open_types.main/2
% CHECK:        X: list.list(int)
% CHECK:        [[@LINE-4]]:9 [1, 2] -> constructor list.'[]'/0{{$}}
% CHECK-NEXT:   [[@LINE-5]]:9 [1, 2] -> constructor list.'[|]'/2{{$}}
% CHECK-NEXT:   [[@LINE-6]]:9 [1, 2] -> constructor list.'[|]'/2{{$}}

    % A tuple has open twins too, its elements open or taking a ground
    % value; a tuple of one is its element.
:- pred tuple_hole(open({int, list(int)})::oa) is det.
tuple_hole(T) :-
    T = {N, [2 | H]},
    N = 1,
    H = [].
% CHECK:      clause at [[@LINE-4]]:1 of pred open_types.tuple_hole/1: pred open_types.tuple_hole(open.open({int, list.list(int)}))
% CHECK-NEXT:   T: open.open({int, list.list(int)})
% CHECK-NEXT:   N: int
% CHECK-NEXT:   H: open.open(list.list(int))
:- pred one(open({list(int)})::oa) is det.
one({[_]}).
% CHECK:      clause at [[@LINE-1]]:1 of pred open_types.one/1: pred open_types.one(open.open({list.list(int)}))
