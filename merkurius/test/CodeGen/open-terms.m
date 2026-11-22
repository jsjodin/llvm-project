% Open terms by the runtime's generic walkers (runtime/open.h;
% docs/INPROGRESS.md, Open types, step 4(a)): to_open and from_open of each
% kind of value, full unification through =, unify_oc, fresh variables that
% a stuck conjunction makes, var and nonvar, and the trail undoing what a
% failed disjunct, if-then-else condition or negation bound. mmc has no open
% types, so the outputs are checked by hand: each line says what it shows.
%
% A collection that meets an open term stops the program with a message,
% until step 5 teaches the collector open layouts.
%
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: not --crash env MERK_GC_EVERY_ALLOC=1 %t 2>&1 | FileCheck %s --check-prefix=COLLECT
% COLLECT: Merkurius runtime: bad root: an open term, which cannot be collected yet ({{.*}})

:- module open_terms.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module bool, char, int8, list, maybe, open, string, uint32.

:- type colour ---> red ; green ; blue.
:- type tree ---> leaf ; node(tree, int, tree).
:- type eight ---> e1(int) ; e2(int) ; e3(int) ; e4(int) ; e5(int)
    ; e6(int) ; e7(int) ; e8(eight).
:- type bundle ---> bundle(closed(list(int)), int).
:- type wrap(T) ---> wrap(T).

    % The value through to_open and from_open.
:- pred round_trip(T::in, io::di, io::uo) is det.
round_trip(X, !IO) :-
    promise_pure ( if impure from_open(to_open(X), Y) then
        io.write(Y, !IO),
        ( if X = Y then io.write_string(" same", !IO) else io.write_string(" differs", !IO) )
    else
        io.write_string("from_open failed", !IO)
    ),
    io.nl(!IO).

:- pred equal(open(T)::ia, open(T)::ia) is semidet.
equal(X, Y) :- X = Y.

:- pred say(string::in, bool::in, io::di, io::uo) is det.
say(What, B, !IO) :-
    io.write_string(What, !IO),
    io.write_string(if B = yes then ": yes\n" else ": no\n", !IO).

    % Binds the hole if it is unbound, or tests it.
:- pred fill(open(list(int))::ia, list(int)::in) is semidet.
fill(Hole, L) :-
    Hole = to_open(L).

:- pred show(string::in, open(list(int))::ia, io::di, io::uo) is det.
show(Name, X, !IO) :-
    promise_pure (
        io.write_string(Name, !IO),
        ( if impure var(X) then
            io.write_string(" is unbound\n", !IO)
        else if impure from_open(X, L) then
            io.write_string(" = ", !IO), io.write(L, !IO), io.nl(!IO)
        else
            io.write_string(" is partly bound\n", !IO)
        )
    ).

    % The first disjunct binds the hole to [1] and then fails, which undoes
    % the binding, so the second can bind it to [3].
:- pred choose(open(list(int))::ia) is cc_nondet.
choose(H) :-
    ( fill(H, [1]), fill(H, [2])
    ; fill(H, [3])
    ).

    % Through !IO, so that it runs after what shows B before it.
:- pred pick(open(list(int))::ia, bool::out, io::di, io::uo) is cc_multi.
pick(H, Chose, !IO) :-
    ( choose(H), Chose = yes ; Chose = no ),
    io.write_string("", !IO).

    % Binds the hole to [1], then fails.
:- pred bind_then_fail(open(list(int))::ia) is semidet.
bind_then_fail(H) :-
    fill(H, [1]), fill(H, [2]).

:- pred holes(io::di, io::uo) is cc_multi.
holes(!IO) :-
    promise_pure (
        % A is a fresh variable, which the stuck conjunction makes.
        show("A", A, !IO),
        ( if fill(A, [4, 5]) then Filled = yes else Filled = no ),
        say("A filled", Filled, !IO),
        show("A", A, !IO),
        ( if fill(A, [4, 5]) then Again = yes else Again = no ),
        say("A unifies with [4, 5] again", Again, !IO),
        ( if fill(A, [6]) then Six = yes else Six = no ),
        say("A unifies with [6]", Six, !IO),
        ( if impure nonvar(A) then NonVar = yes else NonVar = no ),
        say("A is nonvar", NonVar, !IO),

        % The disjunction's first branch's binding is undone.
        show("B", B, !IO),
        pick(B, Chose, !IO),
        say("B chosen", Chose, !IO),
        show("B", B, !IO),

        % A failed condition's binding is undone.
        show("C", C, !IO),
        ( if bind_then_fail(C) then CondOk = yes else CondOk = no ),
        say("the condition succeeded", CondOk, !IO),
        show("C", C, !IO),

        % A negation undoes what its goal bound when the goal succeeds.
        show("D", D, !IO),
        ( if \+ fill(D, [7]) then NotOk = yes else NotOk = no ),
        say("the negation succeeded", NotOk, !IO),
        show("D", D, !IO),

        % from_open fails on an unbound variable.
        show("E", E, !IO),
        ( if impure from_open(E, _) then Closed = yes else Closed = no ),
        say("from_open of an unbound variable", Closed, !IO)
    ).

main(!IO) :-
    % Each kind of value: cells and constants, raw fields boxed or shifted,
    % static boxes and allocated ones, a type that gives tag 7 up, closed(T)
    % boxed and unboxed, and notags and tuples as their arguments.
    round_trip([1, 2, 3], !IO),
    round_trip(node(leaf, 1023, node(leaf, -129, leaf)), !IO),
    round_trip([red, green, blue], !IO),
    round_trip({'x', "text", 2.5, -7i8, 4000000000u32}, !IO),
    round_trip([e1(1), e8(e7(7)), e8(e8(e6(-1)))], !IO),
    round_trip(bundle(closed([1, 2]), 3), !IO),
    round_trip(closed(leaf), !IO),
    round_trip(wrap(yes(wrap(green))), !IO),
    round_trip({[1], 9223372036854775807}, !IO),
    % Unification of ground open terms.
    say("[1, 2] = [1, 2]", (if equal(to_open([1, 2]), to_open([1, 2])) then yes else no), !IO),
    say("[1, 2] = [1, 3]", (if equal(to_open([1, 2]), to_open([1, 3])) then yes else no), !IO),
    say("\"abc\" = \"abc\"", (if equal(to_open("abc"), to_open("abc")) then yes else no), !IO),
    say("2.5 = 2.5", (if equal(to_open(2.5), to_open(2.5)) then yes else no), !IO),
    say("e8(e1(5)) = e8(e1(5))", (if equal(to_open(e8(e1(5))), to_open(e8(e1(5)))) then yes else no), !IO),
    say("e8(e1(5)) = e8(e2(5))", (if equal(to_open(e8(e1(5))), to_open(e8(e2(5)))) then yes else no), !IO),
    say("closed [1, 2] = closed [1, 2]", (if equal(to_open(bundle(closed([1, 2]), 0)), to_open(bundle(closed([1, 2]), 0))) then yes else no), !IO),
    say("closed [1, 2] = closed [2]", (if equal(to_open(bundle(closed([1, 2]), 0)), to_open(bundle(closed([2]), 0))) then yes else no), !IO),
    say("unify_oc 'x' 'x'", (if unify_oc(to_open('x'), to_open('x')) then yes else no), !IO),
    holes(!IO),
    % An open term live at a collection, which stops the program while the
    % collector cannot walk it (the COLLECT run).
    O = to_open([1, 2]),
    pair(3, L),
    io.write(L, !IO),
    promise_pure ( if impure from_open(O, M) then io.write(M, !IO) else true ),
    io.nl(!IO).

:- pred pair(int::in, list(int)::out) is det.
:- pragma no_inline(pair/2).
pair(N, [N, N]).

% OUTPUT:[1, 2, 3] same
% OUTPUT-NEXT:node(leaf, 1023, node(leaf, -129, leaf)) same
% OUTPUT-NEXT:[red, green, blue] same
% OUTPUT-NEXT:{'x', "text", 2.5, -7i8, 4000000000u32} same
% OUTPUT-NEXT:[e1(1), e8(e7(7)), e8(e8(e6(-1)))] same
% OUTPUT-NEXT:bundle(closed([1, 2]), 3) same
% OUTPUT-NEXT:closed(leaf) same
% OUTPUT-NEXT:wrap(yes(wrap(green))) same
% OUTPUT-NEXT:{[1], 9223372036854775807} same
% OUTPUT-NEXT:[1, 2] = [1, 2]: yes
% OUTPUT-NEXT:[1, 2] = [1, 3]: no
% OUTPUT-NEXT:"abc" = "abc": yes
% OUTPUT-NEXT:2.5 = 2.5: yes
% OUTPUT-NEXT:e8(e1(5)) = e8(e1(5)): yes
% OUTPUT-NEXT:e8(e1(5)) = e8(e2(5)): no
% OUTPUT-NEXT:closed [1, 2] = closed [1, 2]: yes
% OUTPUT-NEXT:closed [1, 2] = closed [2]: no
% OUTPUT-NEXT:unify_oc 'x' 'x': yes
% OUTPUT-NEXT:A is unbound
% OUTPUT-NEXT:A filled: yes
% OUTPUT-NEXT:A = [4, 5]
% OUTPUT-NEXT:A unifies with [4, 5] again: yes
% OUTPUT-NEXT:A unifies with [6]: no
% OUTPUT-NEXT:A is nonvar: yes
% OUTPUT-NEXT:B is unbound
% OUTPUT-NEXT:B chosen: yes
% OUTPUT-NEXT:B = [3]
% OUTPUT-NEXT:C is unbound
% OUTPUT-NEXT:the condition succeeded: no
% OUTPUT-NEXT:C is unbound
% OUTPUT-NEXT:D is unbound
% OUTPUT-NEXT:the negation succeeded: no
% OUTPUT-NEXT:D is unbound
% OUTPUT-NEXT:E is unbound
% OUTPUT-NEXT:from_open of an unbound variable: no
% OUTPUT-NEXT:[3, 3][1, 2]
