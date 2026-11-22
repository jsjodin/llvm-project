% open.write of open terms (docs/INPROGRESS.md, Open types, step 6): an
% open term is written as from_open would give it, with each unbound variable
% as _1, _2 and so on, numbered in the order the write meets them, one
% numbering for the whole call: the same variable has the same number,
% wherever it is. Bound variables are their values. A cell already on the
% path from the root, as a cyclic term has, is written as ..., so that the
% write ends. open.string writes the same, and io.write writes a ground
% open term. mmc has no open types, so
% the outputs are checked by hand: each line says what it shows.
%
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module open_write.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, list, open, string.

    % Eight constructors with arguments: the open layout shares tag 6 for
    % the last three, where the ordinary one shares 7 for the last two.
:- type eight ---> p0(int) ; p1(int) ; p2(int) ; p3(int) ; p4(int)
    ; p5(int) ; p6(int) ; p7(int).
:- type item ---> item(string, float, closed(list(int)), eight, char).
:- type wrap ---> wrap(list(int)).
:- type knot ---> knot(knot, int) ; stop.
:- type expr ---> int(int) ; expr + expr ; - expr.

main(!IO) :-
    ground_terms(!IO), holes(!IO), shared(!IO), bound(!IO), cyclic(!IO),
    strings(!IO).

:- pred ground_terms(io::di, io::uo) is det.
ground_terms(!IO) :-
    io.write(to_open([1, 2]), !IO), io.nl(!IO),
    io.write(to_open(item("a", 1.5, closed([1]), p6(6), 'z')), !IO),
    io.nl(!IO),
    io.write(to_open(wrap([3])), !IO), io.nl(!IO),
    io.write(to_open(int(1) + - int(2)), !IO), io.nl(!IO).
% OUTPUT: [1, 2]
% OUTPUT-NEXT: item("a", 1.5, closed([1]), p6(6), 'z')
% OUTPUT-NEXT: wrap([3])
% OUTPUT-NEXT: int(1) + - int(2)

    % Holes of each kind, and an open tail. A notag's word is its
    % argument's, so an unbound open(wrap) is wrap(_1), all it can become.
:- pred holes(io::di, io::uo) is det.
holes(!IO) :-
    promise_pure (
        L = [_, 2 | _] : open(list(int)),
        impure open.write(L, !IO), io.nl(!IO),
        I = item(_, _, _, p7(_), _) : open(item),
        impure open.write(I, !IO), io.nl(!IO),
        W = wrap(_) : open(wrap),
        impure open.write(W, !IO), io.nl(!IO),
        U = _ : open(wrap),
        impure open.write(U, !IO), io.nl(!IO),
        V = _ : open(int),
        impure open.write(V, !IO), io.nl(!IO)
    ).
% OUTPUT-NEXT: [_1, 2 | _2]
% OUTPUT-NEXT: item(_1, _2, _3, p7(_4), _5)
% OUTPUT-NEXT: wrap(_1)
% OUTPUT-NEXT: wrap(_1)
% OUTPUT-NEXT: _1

    % The same variable twice in a term, and in two open values of one call.
:- pred shared(io::di, io::uo) is det.
shared(!IO) :-
    promise_pure (
        L = [A, B, A] : open(list(int)),
        impure open.write(L, !IO), io.nl(!IO),
        M = [B] : open(list(int)),
        impure open.write({L, M}, !IO), io.nl(!IO)
    ).
% OUTPUT-NEXT: [_1, _2, _1]
% OUTPUT-NEXT: {[_1, _2, _1], [_2]}

    % Variables bound, through a chain, are their values.
:- pred bound(io::di, io::uo) is det.
bound(!IO) :-
    promise_pure (
        L = [A, B, C | T] : open(list(int)),
        A = B, B = C, C = 4, T = [],
        impure open.write(L, !IO), io.nl(!IO)
    ).
% OUTPUT-NEXT: [4, 4, 4]

    % Cyclic terms, made by unifying a hole with the term around it, as =
    % has no occurs check: along the tail, and through another field.
:- pred cyclic(io::di, io::uo) is det.
cyclic(!IO) :-
    promise_pure (
        ( if L = [1, 2 | T] : open(list(int)), T = L then
            impure open.write(L, !IO), io.nl(!IO)
        else
            io.write_string("cyclic: no\n", !IO)
        ),
        ( if K = knot(J, 1) : open(knot), J = K then
            impure open.write(K, !IO), io.nl(!IO)
        else
            io.write_string("cyclic: no\n", !IO)
        )
    ).
% OUTPUT-NEXT: [1, 2 | ...]
% OUTPUT-NEXT: knot(..., 1)

:- pred strings(io::di, io::uo) is det.
strings(!IO) :-
    promise_pure (
        L = [1, _, 3] : open(list(int)),
        impure S = open.string(L),
        io.write_string(S, !IO), io.nl(!IO)
    ).
% OUTPUT-NEXT: [1, _1, 3]
