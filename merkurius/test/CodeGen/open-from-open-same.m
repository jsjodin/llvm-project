% from_open of a term whose type has the same open and ordinary layouts
% (design document, "Open terms", from_open; runtime/open.h, kOpenSame):
% if the term has no variable cell, bound or not, the open term is the
% ground one, and from_open returns it without a copy. A notag is its
% argument. A term with an unbound hole fails, and one whose hole was bound
% later is copied.
%
% RUN: merkc -fopen-types -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fopen-types -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module open_from_open_same.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, open.

:- type tree ---> leaf ; node(tree, tree).
:- type w ---> w(tree).

:- pred show(open(T)::ia, io::di, io::uo) is det.
:- pragma no_inline(show/3).
show(X, !IO) :-
    promise_pure (
        if impure from_open(X, G) then io.write(G, !IO)
        else io.write_string("open", !IO)
    ),
    io.nl(!IO).

main(!IO) :-
    show([node(leaf, node(leaf, leaf)), leaf] : open(list(tree)), !IO),
    show(w(node(leaf, leaf)) : open(w), !IO),
    show([node(leaf, _), leaf] : open(list(tree)), !IO),
    show([leaf | _] : open(list(tree)), !IO),
    L = [_, leaf] : open(list(tree)),
    promise_pure ( if L = [node(leaf, leaf) | _] then show(L, !IO) else true ).
% OUTPUT:[node(leaf, node(leaf, leaf)), leaf]
% OUTPUT-NEXT:w(node(leaf, leaf))
% OUTPUT-NEXT:open
% OUTPUT-NEXT:open
% OUTPUT-NEXT:[node(leaf, leaf), leaf]
