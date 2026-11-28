% DESIGN: Open terms (Prolog-style variables) / Operations
% DESIGN: Sized floating point types
% to_open turns a ground term into an open one, and from_open turns it
% back: for scalars, strings, enumerations, lists, tuples, pairs, maybe
% and a type of the program, and through a type variable. from_open fails
% while a hole in the term is unbound and succeeds once it is bound; var
% and nonvar say which a term is. A closed(T) field holds a ground T, and
% a hole of closed(T) can be taken out of a term and filled elsewhere.
% open.string writes a term with holes. It is a template, run for float and
% float32.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/to_from_open_translation.pl
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fopen-types %t.float.d/run_to_from_open_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %t.float.d/run_to_from_open_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:42
% OUTPUT-FLOAT-NEXT:-7
% OUTPUT-FLOAT-NEXT:2.5
% OUTPUT-FLOAT-NEXT:"text"
% OUTPUT-FLOAT-NEXT:green
% OUTPUT-FLOAT-NEXT:[1, 2, 3]
% OUTPUT-FLOAT-NEXT:["x", "y"]
% OUTPUT-FLOAT-NEXT:{1, "one", 1.5}
% OUTPUT-FLOAT-NEXT:3 - "three"
% OUTPUT-FLOAT-NEXT:yes(0.25)
% OUTPUT-FLOAT-NEXT:no
% OUTPUT-FLOAT-NEXT:[circle(1.0), box(2, 3), named("n", blue)]
% OUTPUT-FLOAT-NEXT:[]
% OUTPUT-FLOAT-NEXT:["a", "b", "c"]
% OUTPUT-FLOAT-NEXT:["var", "nonvar", "not var", "nonvar"]
% OUTPUT-FLOAT-NEXT:1 - closed("filled")
% OUTPUT-FLOAT-NEXT:{[1, _1, 3, _1, _2], _2}
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fopen-types %t.float32.d/run_to_from_open_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %t.float32.d/run_to_from_open_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:42
% OUTPUT-FLOAT32-NEXT:-7
% OUTPUT-FLOAT32-NEXT:2.5f32
% OUTPUT-FLOAT32-NEXT:"text"
% OUTPUT-FLOAT32-NEXT:green
% OUTPUT-FLOAT32-NEXT:[1, 2, 3]
% OUTPUT-FLOAT32-NEXT:["x", "y"]
% OUTPUT-FLOAT32-NEXT:{1, "one", 1.5f32}
% OUTPUT-FLOAT32-NEXT:3 - "three"
% OUTPUT-FLOAT32-NEXT:yes(0.25f32)
% OUTPUT-FLOAT32-NEXT:no
% OUTPUT-FLOAT32-NEXT:[circle(1.0f32), box(2, 3), named("n", blue)]
% OUTPUT-FLOAT32-NEXT:[]
% OUTPUT-FLOAT32-NEXT:["a", "b", "c"]
% OUTPUT-FLOAT32-NEXT:["var", "nonvar", "not var", "nonvar"]
% OUTPUT-FLOAT32-NEXT:1 - closed("filled")
% OUTPUT-FLOAT32-NEXT:{[1, _1, 3, _1, _2], _2}
% OUTPUT-FLOAT32-NOT:{{.}}
:- module run_to_from_open.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, maybe, open, pair, string.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type shape
    --->    circle(@T@)
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
    show(2.5@S@, !IO),
    show("text", !IO),
    show(green, !IO),
    show([1, 2, 3], !IO),
    show(["x", "y"], !IO),
    show({1, "one", 1.5@S@}, !IO),
    show(3 - "three", !IO),
    show(yes(0.25@S@), !IO),
    show(no : maybe(int), !IO),
    show([circle(1.0@S@), box(2, 3), named("n", blue)], !IO),
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
