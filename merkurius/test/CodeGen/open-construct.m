% Constructions and get_structure on open terms, compiled inline in the
% open layout (docs/INPROGRESS.md, Open types, step 4(b)). Each open type
% has a merk.type_def of its own, open.open(T), whose fields are all words;
% lowering lays it out with tags 0 to 6, tag 7 being a variable's, and
% shifts an enumeration's constants. The cells the compiled code makes are
% read by the runtime's from_open, and those to_open makes by compiled
% code, so the two layouts must agree: `many` and `eight` share a tag in
% the open layout where the ordinary one has a tag of their own or a
% different shared one. Scalars are converted inline (step 4(d)): shifted
% narrow integers and characters, boxed words. Tuples have open twins too. A get_structure binds an unbound variable to a new
% cell (write mode), or reads a bound one (read mode), and what it binds
% is undone on failure. mmc has no open types, so the outputs are checked
% by hand: each line says what it shows.
%
% RUN: merkc -fc1 -fopen-types -emit-merk %s -o - | FileCheck %s --check-prefix=MERK
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK-DAG: merk.type_def "open.open(open_construct.eight)" [#merk.constructor<"p0"(!merk.word)>, {{.*}}] open
% MERK-DAG: merk.type_def "open.open(list.list(int))" [#merk.constructor<"[]">, #merk.constructor<"[|]"(!merk.word, !merk.word)>] open

:- module open_construct.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, list, open.

:- type tree ---> leaf ; node(tree, int, tree).
:- type colour ---> red ; green ; blue.
    % Seven constructors with arguments and a constant: ordinary tags 1 to
    % 7, open tags 1 to 5 and a shared 6 for the last two.
:- type many ---> none ; a(int) ; b(int) ; c(int) ; d(int) ; e(int)
    ; f(int) ; g(int).
    % Eight with arguments: ordinary tags 0 to 6 and a shared 7 for the
    % last two; open tags 0 to 5 and a shared 6 for the last three.
:- type eight ---> p0(int) ; p1(int) ; p2(int) ; p3(int) ; p4(int)
    ; p5(int) ; p6(int) ; p7(int).
:- type wrap ---> wrap(list(int)).
:- type na ---> na(int8, uint8, int16, uint16, int32, uint32).
:- type nb ---> nb(int64, uint64, uint, int, float, character).
:- type nc ---> nc(string, int).
:- type pt ---> pt(int, colour, string).
:- type bundle ---> bundle(closed(list(int)), float, character).

:- pred show(open(T)::ia, io::di, io::uo) is det.
show(X, !IO) :-
    promise_pure (
        if impure from_open(X, G) then io.write(G, !IO)
        else io.write_string("open", !IO)
    ),
    io.nl(!IO).

:- pred yes_no(bool::in, io::di, io::uo) is det.
yes_no(yes, !IO) :- io.write_string("yes\n", !IO).
yes_no(no, !IO) :- io.write_string("no\n", !IO).

    % Constructions, with ground arguments through to_open.
:- pred mk1(open(list(int))::oa) is det.
mk1([1, 2, 3]).
:- pred mk2(open(list(int))::oa) is det.
mk2([7 | _]).
:- pred mk3(open(tree)::oa) is det.
mk3(node(leaf, 3, node(_, 4, leaf))).
:- pred mk4(open(list(colour))::oa) is det.
mk4([red, blue, _]).

    % get_structure: the 1 is unified with the head, which binds an unbound
    % one; H is taken out of the head, which fails if it is unbound.
:- pred head_one(open(list(int))::ia) is semidet.
head_one([1 | _]).
:- pred head_out(open(list(int))::ia, int::out) is semidet.
head_out([H | _], H).
:- pred second(open(list(int))::ia, open(list(int))::oa) is semidet.
second([_, X | _], [X]).

:- pred test_a(open(list(int))::ia, io::di, io::uo) is det.
test_a(A, !IO) :-
    promise_pure ( if head_out(A, H) then io.write_int(H, !IO), io.nl(!IO)
        else io.write_string("no\n", !IO) ),
    promise_pure ( if head_one(A) then R = yes else R = no ), yes_no(R, !IO),
    promise_pure ( if second(A, S) then show(S, !IO)
        else io.write_string("no\n", !IO) ).

:- pred test_b(open(list(int))::ia, io::di, io::uo) is det.
test_b(B, !IO) :-
    promise_pure ( if head_one(B) then R = yes else R = no ), yes_no(R, !IO),
    promise_pure ( if head_out(B, H) then io.write_int(H, !IO), io.nl(!IO)
        else io.write_string("no\n", !IO) ).

    % E is a fresh variable, which head_one binds to [1 | _] (write mode).
:- pred test_e(io::di, io::uo) is det.
test_e(!IO) :-
    promise_pure ( if head_one(E), E = [_], impure from_open(E, G) then
        io.write(G, !IO), io.nl(!IO)
    else
        io.write_string("no\n", !IO)
    ).

    % Holes filled by get_structure in a condition.
:- pred test_d(open(list(colour))::ia, io::di, io::uo) is det.
test_d(D, !IO) :-
    promise_pure ( if D = [_, _, green] then show(D, !IO)
        else io.write_string("no\n", !IO) ).
:- pred test_c(open(tree)::ia, io::di, io::uo) is det.
test_c(C, !IO) :-
    promise_pure ( if C = node(_, _, node(leaf, _, _)) then show(C, !IO)
        else io.write_string("no\n", !IO) ).

    % Undoing: try binds X's tail and fails, and so does the first
    % disjunct.
:- pred try(open(list(int))::ia) is failure.
try(X) :- X = [_, 2 | T], T = [], fail.
:- pred fill(open(list(int))::ia) is semidet.
fill(X) :- X = [_, 2 | T], T = [3].
:- pred mk_one(open(list(int))::oa) is det.
mk_one([1 | _]).

:- pred test_undo(io::di, io::uo) is det.
test_undo(!IO) :-
    mk_one(X),
    promise_pure ( if try(X) then true else true ), show(X, !IO),
    promise_pure ( if ( X = [9 | _], fail ; X = [_, _ | []] ) then show(X, !IO)
        else io.write_string("no\n", !IO) ),
    mk_one(Y),
    promise_pure ( if fill(Y) then show(Y, !IO)
        else io.write_string("no\n", !IO) ),
    promise_pure ( if Y = [_, _, 4] then show(Y, !IO)
        else io.write_string("clash\n", !IO) ).

    % The layouts' tags, both ways.
:- pred manys(list(open(many))::oa) is det.
manys([none, a(1), b(2), c(3), d(4), e(5), f(6), g(7)]).
:- pred eights(list(open(eight))::oa) is det.
eights([p0(0), p1(1), p2(2), p3(3), p4(4), p5(5), p6(6), p7(7)]).

:- pred which_many(open(many)::ia, int::out) is semidet.
which_many(X, N) :-
    promise_pure ( if X = none then N = 0 else if X = a(A) then N = A
    else if X = f(F) then N = F else X = g(N) ).
:- pred which_eight(open(eight)::ia, int::out) is semidet.
which_eight(X, N) :-
    promise_pure ( if X = p5(A) then N = A else if X = p6(B) then N = B
    else X = p7(N) ).

:- pred w_many(open(many)::ia, io::di, io::uo) is det.
w_many(X, !IO) :-
    promise_pure ( if which_many(X, N) then io.write_int(N, !IO)
        else io.write_string("-", !IO) ),
    io.write_string(" ", !IO).
:- pred w_eight(open(eight)::ia, io::di, io::uo) is det.
w_eight(X, !IO) :-
    promise_pure ( if which_eight(X, N) then io.write_int(N, !IO)
        else io.write_string("-", !IO) ),
    io.write_string(" ", !IO).

:- pred each(pred(T, io, io), list(T), io, io).
:- mode each(pred(ia, di, uo) is det, ia, di, uo) is det.
each(_, [], !IO).
each(P, [X | Xs], !IO) :- P(X, !IO), each(P, Xs, !IO).

    % A notag, shifted and boxed fields, and a closed(T) argument boxed.
:- pred others(open(wrap)::oa, open(pt)::oa, open(bundle)::oa) is det.
others(wrap([1, 2]), pt(3, blue, "s"), bundle(closed([4]), 2.5, 'x')).

:- pred tuples(open({int, colour, string})::oa,
    open({int, list(int)})::oa, open({list(int)})::oa) is det.
tuples({3, blue, "s"}, {1, [2 | _]}, {[1 | _]}).

:- pred test_tuples(io::di, io::uo) is det.
test_tuples(!IO) :-
    tuples(T3, T2, T1),
    show(T3, !IO), show(T2, !IO), show(T1, !IO),
    promise_pure ( if T3 = {_, C, _}, T2 = {_, [_, 3]}, T1 = {[_]} then
        show(C, !IO), show(T2, !IO), show(T1, !IO)
    else
        io.write_string("no\n", !IO)
    ).

:- pred read_others(open(wrap)::ia, open(pt)::ia, io::di, io::uo) is det.
read_others(W, P, !IO) :-
    promise_pure ( if W = wrap([_, X]), P = pt(_, C, S) then
        show(X, !IO), show(C, !IO), show(S, !IO)
    else
        io.write_string("no\n", !IO)
    ).

    % Each scalar kind, at the edges of its range, built inline.
:- pred mk_nums(open(na)::oa, open(nb)::oa, open(nc)::oa) is det.
mk_nums(na(-128i8, 255u8, -32768i16, 65535u16, -2147483648i32, 4294967295u32),
    nb(-9223372036854775807i64, 18446744073709551615u64, 7u, -1, -0.5, 'z'),
    nc("", 1)).

    % Read into variables of ground types, by compiled code.
:- pred get_a(open(na)::ia, int8::out, uint8::out, int16::out, uint16::out,
    int32::out, uint32::out) is semidet.
get_a(na(A, B, C, D, E, F), A, B, C, D, E, F).
:- pred get_b(open(nb)::ia, int64::out, uint64::out, uint::out, int::out,
    float::out, character::out) is semidet.
get_b(nb(A, B, C, D, E, F), A, B, C, D, E, F).
:- pred get_c(open(nc)::ia, string::out, int::out) is semidet.
get_c(nc(A, B), A, B).

:- pred read_nums(open(na)::ia, open(nb)::ia, open(nc)::ia,
    io::di, io::uo) is det.
read_nums(Na, Nb, Nc, !IO) :-
    promise_pure ( if get_a(Na, A, B, C, D, E, F) then
        io.write({A, B, C, D, E, F}, !IO), io.nl(!IO)
    else
        io.write_string("no\n", !IO)
    ),
    promise_pure ( if get_b(Nb, G, H, I, J, K, L) then
        io.write({G, H, I, J, K, L}, !IO), io.nl(!IO)
    else
        io.write_string("no\n", !IO)
    ),
    promise_pure ( if get_c(Nc, M, N) then
        io.write({M, N}, !IO), io.nl(!IO)
    else
        io.write_string("no\n", !IO)
    ).

main(!IO) :-
    mk_nums(Na, Nb, Nc), show(Na, !IO), show(Nb, !IO), show(Nc, !IO),
    read_nums(Na, Nb, Nc, !IO),
    read_nums(to_open(na(1i8, 2u8, 3i16, 4u16, 5i32, 6u32)),
        to_open(nb(7i64, 8u64, 9u, 10, 11.5, 'a')), to_open(nc("b", 12)), !IO),
    mk1(A), show(A, !IO),
    mk2(B), show(B, !IO),
    mk3(C), show(C, !IO),
    mk4(D), show(D, !IO),
    test_a(A, !IO), test_b(B, !IO), test_e(!IO),
    test_d(D, !IO), test_c(C, !IO),
    test_undo(!IO),
    manys(Ms), each(show, Ms, !IO), each(w_many, Ms, !IO), io.nl(!IO),
    eights(Es), each(show, Es, !IO), each(w_eight, Es, !IO), io.nl(!IO),
    RMs = list.map(to_open, [none, a(1), b(2), c(3), d(4), e(5), f(6), g(7)]),
    each(w_many, RMs, !IO), io.nl(!IO),
    REs = list.map(to_open, [p0(0), p1(1), p2(2), p3(3), p4(4), p5(5),
        p6(6), p7(7)]),
    each(w_eight, REs, !IO), io.nl(!IO),
    others(W, P, Bu), show(W, !IO), show(P, !IO), show(Bu, !IO),
    read_others(W, P, !IO),
    read_others(to_open(wrap([5, 6])), to_open(pt(7, green, "t")), !IO),
    test_tuples(!IO).

% Scalars built inline, read by from_open, then by compiled code; to_open's
% read by compiled code.
% OUTPUT:na(-128i8, 255u8, -32768i16, 65535u16, -2147483648i32, 4294967295u32)
% OUTPUT-NEXT:nb(-9223372036854775807i64, 18446744073709551615u64, 7u, -1, -0.5, 'z')
% OUTPUT-NEXT:nc("", 1)
% OUTPUT-NEXT:{-128i8, 255u8, -32768i16, 65535u16, -2147483648i32, 4294967295u32}
% OUTPUT-NEXT:{-9223372036854775807i64, 18446744073709551615u64, 7u, -1, -0.5, 'z'}
% OUTPUT-NEXT:{"", 1}
% OUTPUT-NEXT:{1i8, 2u8, 3i16, 4u16, 5i32, 6u32}
% OUTPUT-NEXT:{7i64, 8u64, 9u, 10, 11.5, 'a'}
% OUTPUT-NEXT:{"b", 12}
% Constructions: a ground one, and ones with holes.
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:open
% OUTPUT-NEXT:open
% OUTPUT-NEXT:open
% test_a: the head taken out, matched by 1, and the second element.
% OUTPUT-NEXT:1
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:[2]
% test_b: [7 | _] does not match [1 | _]; its head is 7.
% OUTPUT-NEXT:no
% OUTPUT-NEXT:7
% test_e: a fresh variable bound by head_one, then its tail by E = [_].
% OUTPUT-NEXT:[1]
% test_d and test_c: holes filled in a condition.
% OUTPUT-NEXT:[red, blue, green]
% OUTPUT-NEXT:node(leaf, 3, node(leaf, 4, leaf))
% test_undo: try's binding undone; the second disjunct's tail; fill; a clash.
% OUTPUT-NEXT:open
% OUTPUT-NEXT:open
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:clash
% Compiled cells of many and eight, read by from_open and by compiled code.
% OUTPUT-NEXT:none
% OUTPUT-NEXT:a(1)
% OUTPUT-NEXT:b(2)
% OUTPUT-NEXT:c(3)
% OUTPUT-NEXT:d(4)
% OUTPUT-NEXT:e(5)
% OUTPUT-NEXT:f(6)
% OUTPUT-NEXT:g(7)
% OUTPUT-NEXT:0 1 - - - - 6 7 
% OUTPUT-NEXT:p0(0)
% OUTPUT-NEXT:p1(1)
% OUTPUT-NEXT:p2(2)
% OUTPUT-NEXT:p3(3)
% OUTPUT-NEXT:p4(4)
% OUTPUT-NEXT:p5(5)
% OUTPUT-NEXT:p6(6)
% OUTPUT-NEXT:p7(7)
% OUTPUT-NEXT:- - - - - 5 6 7 
% to_open's cells read by compiled code.
% OUTPUT-NEXT:0 1 - - - - 6 7 
% OUTPUT-NEXT:- - - - - 5 6 7 
% A notag, shifted and boxed fields, and a boxed closed(T).
% OUTPUT-NEXT:wrap([1, 2])
% OUTPUT-NEXT:pt(3, blue, "s")
% OUTPUT-NEXT:bundle(closed([4]), 2.5, 'x')
% Read by compiled code: compiled cells, then to_open's.
% OUTPUT-NEXT:2
% OUTPUT-NEXT:blue
% OUTPUT-NEXT:"s"
% OUTPUT-NEXT:6
% OUTPUT-NEXT:green
% OUTPUT-NEXT:"t"
% Tuples, and a tuple of one, which is its element; then holes filled.
% OUTPUT-NEXT:{3, blue, "s"}
% OUTPUT-NEXT:open
% OUTPUT-NEXT:open
% OUTPUT-NEXT:blue
% OUTPUT-NEXT:{1, [2, 3]}
% OUTPUT-NEXT:{ [1] }
