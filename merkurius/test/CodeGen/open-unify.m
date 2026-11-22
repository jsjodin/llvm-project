% A unify procedure per open type (docs/INPROGRESS.md, Open types, step
% 4(d) part 3): X = Y on a ground open type calls unify_open.<type>, which
% dereferences, binds an unbound side to the other, and unifies each field
% inline for a scalar or by its type's procedure, the last by a tail call. A
% notag is its argument, so wrap has none. A cycle along last fields is
% found by Brent's method; one through other fields goes past the depth
% bound to the runtime's walker, which polymorphic code calls too.
%
% RUN: merkc -fc1 -fopen-types -emit-merk %s -o - | FileCheck %s
% RUN: merkc -fopen-types %s -o %t && env MERK_HEAP_CHUNK_BYTES=134217728 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && env MERK_HEAP_CHUNK_BYTES=134217728 %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines

% CHECK-LABEL: merk.proc {{.*}}@open_unify.eq_2_0(
% CHECK: merk_open_unify
% CHECK-LABEL: merk.proc {{.*}}@open_unify.rec_eq_2_0(
% CHECK: merk.call @unify_open.open_unify.rec(
% CHECK-NOT: unify_open.open_unify.wrap
% CHECK: merk.proc private @unify_open.open_unify.rec(
% CHECK: merk.call @"unify_open.list.list(int)"(

:- module open_unify.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, float, int, int8, list, open, string.

:- type colour ---> red ; green ; blue.
:- type rec ---> rec(int8, colour, float, string, closed(list(int)), list(int)).
:- type wrap ---> wrap(rec).
:- type knot ---> knot(knot, int) ; stop.

:- pred t(string::in, bool::in, io::di, io::uo) is det.
t(S, B, !IO) :-
    io.write_string(S, !IO),
    io.write_string(if B = yes then ": yes\n" else ": no\n", !IO).

:- pred eq(open(T)::ia, open(T)::ia) is semidet.
:- pragma no_inline(eq/2).
eq(X, Y) :- X = Y.

:- pred rec_eq(open(wrap)::ia, open(wrap)::ia) is semidet.
:- pragma no_inline(rec_eq/2).
rec_eq(X, Y) :- X = Y.

:- pred list_eq(open(list(int))::ia, open(list(int))::ia) is semidet.
:- pragma no_inline(list_eq/2).
list_eq(X, Y) :- X = Y.

:- pred knot_eq(open(knot)::ia, open(knot)::ia) is semidet.
:- pragma no_inline(knot_eq/2).
knot_eq(X, Y) :- X = Y.

:- func r(int8, colour, float, string, list(int), list(int)) = open(wrap).
r(A, B, C, D, E, F) = wrap(rec(A, B, C, D, closed(E), F)).

main(!IO) :-
    fields(!IO), holes(!IO), undo(!IO), cyclic(!IO), poly(!IO), long(!IO).

:- pred fields(io::di, io::uo) is det.
fields(!IO) :-
    R = r(1i8, green, 2.5, "s", [1, 2], [3]),
    t("same record", (if rec_eq(R, r(1i8, green, 2.5, "s", [1, 2], [3])) then yes else no), !IO),
    t("int8 differs", (if rec_eq(R, r(2i8, green, 2.5, "s", [1, 2], [3])) then yes else no), !IO),
    t("enum differs", (if rec_eq(R, r(1i8, blue, 2.5, "s", [1, 2], [3])) then yes else no), !IO),
    t("float differs", (if rec_eq(R, r(1i8, green, 2.0, "s", [1, 2], [3])) then yes else no), !IO),
    t("string differs", (if rec_eq(R, r(1i8, green, 2.5, "t", [1, 2], [3])) then yes else no), !IO),
    t("closed differs", (if rec_eq(R, r(1i8, green, 2.5, "s", [1], [3])) then yes else no), !IO),
    t("list differs", (if rec_eq(R, r(1i8, green, 2.5, "s", [1, 2], [4])) then yes else no), !IO).

    % Holes of every kind, bound to the other side's values.
:- pred holes(io::di, io::uo) is det.
holes(!IO) :-
    R = r(1i8, green, 2.5, "s", [1, 2], [3]),
    promise_pure ( if
        H = wrap(rec(_, _, _, _, _, [_ | _])),
        rec_eq(H, R),
        from_open(H, G)
    then
        io.write(G, !IO), io.nl(!IO)
    else
        io.write_string("holes: no\n", !IO)
    ).

    % A failed unification undoes what it bound.
:- pred undo(io::di, io::uo) is det.
undo(!IO) :-
    promise_pure (
        L = [A, 2] : open(list(int)),
        ( if list_eq(L, [1, 3]) then
            io.write_string("undo: unified\n", !IO)
        else if impure var(A) then
            io.write_string("undo: A unbound\n", !IO)
        else
            io.write_string("undo: A bound\n", !IO)
        )
    ).

    % Cyclic terms: along the tail, with cycles of different lengths, and
    % through a field other than the last.
:- pred cyclic(io::di, io::uo) is det.
cyclic(!IO) :-
    promise_pure (
        ( if L1 = [1 | L1], L2 = [1, 1 | L2], list_eq(L1, L2) then
            io.write_string("cyclic lists: yes\n", !IO)
        else
            io.write_string("cyclic lists: no\n", !IO)
        ),
        ( if M1 = [1 | M1], M2 = [1, 2 | M2], list_eq(M1, M2) then
            io.write_string("unequal cyclic lists: yes\n", !IO)
        else
            io.write_string("unequal cyclic lists: no\n", !IO)
        ),
        ( if K1 = knot(K1, 1), K2 = knot(knot(K2, 1), 1), knot_eq(K1, K2) then
            io.write_string("cyclic knots: yes\n", !IO)
        else
            io.write_string("cyclic knots: no\n", !IO)
        )
    ).

    % Polymorphic: the runtime's walker.
:- pred poly(io::di, io::uo) is det.
poly(!IO) :-
    t("polymorphic", (if eq([1, 2] : open(list(int)), [1, 2]) then yes else no), !IO),
    t("polymorphic differs", (if eq([1, 2] : open(list(int)), [1, 3]) then yes else no), !IO).

    % Long lists, one with holes, unified in constant stack, at -O0 too.
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

:- pred total(open(list(int))::ia, int::in, int::out) is semidet.
total(L, A0, A) :-
    promise_pure ( if L = [X | T] then
        total(T, A0 + X, A)
    else
        A = A0
    ).

:- pred long(io::di, io::uo) is det.
long(!IO) :-
    promise_pure ( if
        build(1, 200000, 200001, Full), build(1, 200000, 1000, Holes),
        list_eq(Holes, Full), total(Holes, 0, S)
    then
        io.write_int(S, !IO), io.nl(!IO)
    else
        io.write_string("long: no\n", !IO)
    ).

% OUTPUT:same record: yes
% OUTPUT-NEXT:int8 differs: no
% OUTPUT-NEXT:enum differs: no
% OUTPUT-NEXT:float differs: no
% OUTPUT-NEXT:string differs: no
% OUTPUT-NEXT:closed differs: no
% OUTPUT-NEXT:list differs: no
% OUTPUT-NEXT:wrap(rec(1i8, green, 2.5, "s", closed([1, 2]), [3]))
% OUTPUT-NEXT:undo: A unbound
% OUTPUT-NEXT:cyclic lists: yes
% OUTPUT-NEXT:unequal cyclic lists: no
% OUTPUT-NEXT:cyclic knots: yes
% OUTPUT-NEXT:polymorphic: yes
% OUTPUT-NEXT:polymorphic differs: no
% OUTPUT-NEXT:20000100000
