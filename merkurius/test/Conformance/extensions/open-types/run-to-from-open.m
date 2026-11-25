% DESIGN: Open terms (Prolog-style variables) / Operations
% to_open turns a ground term into an open one, and from_open turns it
% back: for scalars, strings, enumerations, lists, tuples, pairs, maybe
% and a type of the program, and through a type variable. from_open fails
% while a hole in the term is unbound and succeeds once it is bound; var
% and nonvar say which a term is. A closed(T) field holds a ground T, and
% a hole of closed(T) can be taken out of a term and filled elsewhere.
% open.string writes a term with holes.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/to_from_open_translation.pl
%
% RUN: merkc -O0 -fopen-types %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:42
% OUTPUT-NEXT:-7
% OUTPUT-NEXT:2.5
% OUTPUT-NEXT:"text"
% OUTPUT-NEXT:green
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:["x", "y"]
% OUTPUT-NEXT:{1, "one", 1.5}
% OUTPUT-NEXT:3 - "three"
% OUTPUT-NEXT:yes(0.25)
% OUTPUT-NEXT:no
% OUTPUT-NEXT:[circle(1.0), box(2, 3), named("n", blue)]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:["a", "b", "c"]
% OUTPUT-NEXT:["var", "nonvar", "not var", "nonvar"]
% OUTPUT-NEXT:1 - closed("filled")
% OUTPUT-NEXT:{[1, _1, 3, _1, _2], _2}
% OUTPUT-NOT:{{.}}
:- module run_to_from_open.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list, maybe, open, pair, string.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type shape
    --->    circle(float)
    ;       box(int, int)
    ;       named(string, colour).

    % Through a type variable: the walkers take T's type_info.
:- pred round_trip(T::in, T::out) is semidet.
:- pragma promise_pure(round_trip/2).

round_trip(X, Y) :-
    impure from_open(to_open(X), Y).

:- pred show(T::in, io::di, io::uo) is det.

show(X, !IO) :-
    ( if round_trip(X, Y) then
        io.write_line(Y, !IO)
    else
        io.write_string("from_open failed\n", !IO)
    ).

    % A list with a hole at its end: from_open fails, then the hole is
    % bound and it succeeds.
:- pred fill_later(list(string)::out, list(string)::out) is semidet.
:- pragma promise_pure(fill_later/2).

fill_later(Before, After) :-
    L = [to_open("a"), to_open("b") | T],
    impure try_close(L, Before),
    T = to_open(["c"]),
    impure from_open(L, After).

    % Impure, so that it stays before the goal that fills the hole.
:- impure pred try_close(open(list(string))::ia, list(string)::out) is det.

try_close(L, R) :-
    promise_impure (
        if impure from_open(L, B) then R = B else R = []
    ).

    % var and nonvar of a hole, before and after it is bound.
:- pred var_nonvar(list(string)::out) is det.

var_nonvar(Answers) :-
    X = (to_open(1) - H) : open(pair(int, int)),
    promise_pure (
        ( if impure var(H) then A1 = "var" else A1 = "not var" ),
        ( if impure nonvar(X) then A2 = "nonvar" else A2 = "not nonvar" ),
        ( if H = to_open(2) then true else true ),
        ( if impure var(H) then A3 = "var" else A3 = "not var" ),
        ( if impure nonvar(H) then A4 = "nonvar" else A4 = "not nonvar" )
    ),
    Answers = [A1, A2, A3, A4].

    % A hole of closed(string) taken out of the pair and filled by bind.
:- pred closed_hole(pair(int, closed(string))::out) is semidet.
:- pragma promise_pure(closed_hole/1).

closed_hole(P) :-
    X = (to_open(1) - H) : open(pair(int, closed(string))),
    bind(H, "filled"),
    impure from_open(X, P).

:- pred bind(closed(string)::ia, string::in) is semidet.

bind(H, S) :-
    H = closed(S).

:- pred written(string::out) is det.
:- pragma promise_pure(written/1).

written(S) :-
    L = [to_open(1), A, to_open(3), A, B] : open(list(int)),
    impure S = open.string({L, B}).

main(!IO) :-
    show(42, !IO),
    show(-7, !IO),
    show(2.5, !IO),
    show("text", !IO),
    show(green, !IO),
    show([1, 2, 3], !IO),
    show(["x", "y"], !IO),
    show({1, "one", 1.5}, !IO),
    show(3 - "three", !IO),
    show(yes(0.25), !IO),
    show(no : maybe(int), !IO),
    show([circle(1.0), box(2, 3), named("n", blue)], !IO),
    ( if fill_later(Before, After) then
        io.write_line(Before, !IO),
        io.write_line(After, !IO)
    else
        io.write_string("fill_later failed\n", !IO)
    ),
    var_nonvar(Answers),
    io.write_line(Answers, !IO),
    ( if closed_hole(P) then
        io.write_line(P, !IO)
    else
        io.write_string("closed_hole failed\n", !IO)
    ),
    written(S),
    io.write_string(S, !IO),
    io.nl(!IO).
