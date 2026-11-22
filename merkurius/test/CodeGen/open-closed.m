% closed(T) values that may be unbound (docs/INPROGRESS.md, Open types,
% step 4(c)). A closed(T) variable whose inst may have `any` in it holds
% closed(T)'s open word, a box or a variable's reference, and so does a
% parameter whose mode has `any` in it; a ground one holds T's word, at no
% cost. Values are boxed where they meet an open word and unboxed where they
% must be T's: at calls, at the exit, and in ordinary terms. So a hole of
% type closed(T) can be taken out of an open term, passed around, filled
% elsewhere, and unified with another. mmc has no open types, so the
% outputs are checked by hand: each line says what it shows.
%
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module open_closed.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, open, pair.

:- type big ---> big(int, int).
:- type bundle ---> bundle(closed(big), closed(big), int).
:- type single ---> single(closed(big), int).

:- pred show(open(T)::ia, io::di, io::uo) is det.
show(X, !IO) :-
    promise_pure (
        if from_open(X, G) then io.write(G, !IO)
        else io.write_string("open", !IO)
    ),
    io.nl(!IO).

:- pred holes(open(bundle)::oa) is det.
holes(bundle(_, _, 1)).

    % The design document's example: Hole = closed(Chunk).
:- pred fill_first(open(bundle)::ia, big::in) is semidet.
fill_first(B, Chunk) :-
    B = bundle(Hole, _, _),
    Hole = closed(Chunk).

:- pred second_hole(open(bundle)::ia, closed(big)::oa) is det.
second_hole(bundle(_, H, _), H).
:- pred fill(closed(big)::ia, big::in) is semidet.
fill(H, C) :- H = closed(C).
:- pred get(closed(big)::ia, big::out) is semidet.
get(closed(C), C).
:- pred first(open(bundle)::ia, big::out) is semidet.
first(bundle(closed(C), _, _), C).
:- pred same(closed(big)::ia, closed(big)::ia) is semidet.
same(X, X).

    % An oa output built from a ground value, boxed at the exit.
:- pred mk(closed(big)::oa) is det.
mk(closed(big(1, 2))).

:- pred hold(closed(big)::ia) is det.
hold(_).
:- pred use(closed(big)::in, io::di, io::uo) is det.
use(closed(B), !IO) :- io.write(B, !IO), io.nl(!IO).

:- pred holes_test(io::di, io::uo) is det.
holes_test(!IO) :-
    holes(B), show(B, !IO),
    promise_pure ( if first(B, C0) then io.write(C0, !IO), io.nl(!IO)
        else io.write_string("first unbound\n", !IO) ),
    promise_pure ( if fill_first(B, big(1, 2)) then show(B, !IO)
        else io.write_string("no\n", !IO) ),
    promise_pure ( if first(B, C) then io.write(C, !IO), io.nl(!IO)
        else io.write_string("no\n", !IO) ),
    promise_pure ( if fill_first(B, big(9, 9)) then io.write_string("no\n", !IO)
        else io.write_string("clash\n", !IO) ),
    second_hole(B, H),
    promise_pure ( if fill(H, big(3, 4)) then show(B, !IO)
        else io.write_string("no\n", !IO) ),
    holes(B2),
    promise_pure ( if B2 = bundle(X, Y, _), same(X, Y), fill(X, big(5, 6)) then
        show(B2, !IO)
    else
        io.write_string("no\n", !IO)
    ).

:- pred conversions(io::di, io::uo) is det.
conversions(!IO) :-
    mk(M),
    promise_pure ( if get(M, C) then io.write(C, !IO)
        else io.write_string("no", !IO) ), io.nl(!IO),
    % A ground closed(big) given to ia parameters: boxed at the call.
    G = closed(big(3, 4)),
    promise_pure ( if fill(G, big(3, 4)) then io.write_string("same\n", !IO)
        else io.write_string("no\n", !IO) ),
    promise_pure ( if fill(G, big(0, 0)) then io.write_string("no\n", !IO)
        else io.write_string("clash\n", !IO) ),
    % A fresh hole filled, then given to an `in` parameter and put in
    % ordinary and open terms: unboxed, and boxed again.
    hold(F),
    promise_pure ( if F = closed(big(5, 6)) then
        use(F, !IO),
        io.write([F], !IO), io.nl(!IO),
        show(to_open(single(F, 7)), !IO)
    else
        io.write_string("no\n", !IO)
    ),
    % An ordinary type with open and closed parts, unified by its own
    % unify entry.
    A = to_open(1) - closed(2),
    promise_pure ( if same_pair(A, _ - closed(2)) then
        io.write_string("pair\n", !IO)
    else
        io.write_string("no\n", !IO)
    ).

:- pred same_pair(pair(open(int), closed(int))::ia,
    pair(open(int), closed(int))::ia) is semidet.
same_pair(X, Y) :- X = Y.

main(!IO) :-
    holes_test(!IO),
    conversions(!IO).

% Two holes, the first unbound, then filled by Hole = closed(Chunk).
% OUTPUT:open
% OUTPUT-NEXT:first unbound
% OUTPUT-NEXT:open
% OUTPUT-NEXT:big(1, 2)
% OUTPUT-NEXT:clash
% The second hole taken out, and filled elsewhere.
% OUTPUT-NEXT:bundle(closed(big(1, 2)), closed(big(3, 4)), 1)
% Two holes unified, then one filled.
% OUTPUT-NEXT:bundle(closed(big(5, 6)), closed(big(5, 6)), 1)
% conversions: an oa output, ground values at ia parameters, a filled hole.
% OUTPUT-NEXT:big(1, 2)
% OUTPUT-NEXT:same
% OUTPUT-NEXT:clash
% OUTPUT-NEXT:big(5, 6)
% OUTPUT-NEXT:[closed(big(5, 6))]
% OUTPUT-NEXT:single(closed(big(5, 6)), 7)
% OUTPUT-NEXT:pair
