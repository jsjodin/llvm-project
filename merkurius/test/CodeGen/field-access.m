% Field accesses and updates, which no module defines, are the unifications
% the manual defines them by (reference manual, "Field access functions"):
% the term deconstructed by the field's constructor, tested first when the
% type has others, and for an update constructed again with the new value
% in the field's place. Of one constructor and of several, polymorphic, by
% state variables and nested. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module field_access.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list, maybe, string.

    % One constructor: its fields' functions are det.
:- type point
    --->    point(
                x :: int,
                y :: int,
                label :: string,
                weight :: float
            ).

    % Several: those of a field only one constructor has are semidet.
:- type shape
    --->    circle(radius :: float)
    ;       rect(width :: int, height :: int)
    ;       empty.

    % Polymorphic, with a field whose type is the parameter.
:- type box(T)
    --->    box(contents :: T, count :: int).

:- type nested
    --->    nested(inner :: point, tag :: maybe(int)).

:- func area(shape) = int is semidet.
area(S) = S ^ width * S ^ height.

:- func bump(box(T)) = box(T).
bump(B) = B ^ count := B ^ count + 1.

:- func put(box(T), T) = box(T).
put(B, X) = B ^ contents := X.

:- pred show(T::in, io::di, io::uo) is det.
show(X, !IO) :-
    io.write(X, !IO),
    io.nl(!IO).

main(!IO) :-
    P0 = point(1, 2, "p", 0.5),
    show(P0 ^ x + P0 ^ y, !IO),
    show(P0 ^ label, !IO),
    show(P0 ^ weight, !IO),
    P1 = P0 ^ x := 10,
    P2 = P1 ^ label := "q",
    show(P2, !IO),
    show(P0, !IO),
    % State variable field updates.
    some [!P] (
        !:P = P2,
        !P ^ y := 20,
        !P ^ weight := !.P ^ weight * 4.0,
        show(!.P, !IO)
    ),
    % Semidet: the field's constructor or not.
    ( if A = area(rect(3, 4)) then show(A, !IO) else show("no area", !IO) ),
    ( if A2 = area(circle(1.0)) then show(A2, !IO) else show("no area", !IO) ),
    ( if R = circle(2.5) ^ radius then show(R, !IO) else show("no radius", !IO) ),
    ( if R2 = empty ^ radius then show(R2, !IO) else show("no radius", !IO) ),
    ( if S = (rect(1, 2) ^ height := 7) then show(S, !IO) else show("no", !IO) ),
    ( if S2 = (empty ^ height := 7) then show(S2, !IO) else show("no", !IO) ),
    % Polymorphic.
    B0 = box("s", 0),
    show(bump(bump(B0)), !IO),
    show(put(B0, "t"), !IO),
    show(bump(box([1, 2], 5)) ^ contents, !IO),
    % Nested fields.
    N0 = nested(P0, yes(3)),
    show(N0 ^ inner ^ label, !IO),
    N1 = N0 ^ inner ^ x := 99,
    show(N1, !IO),
    N2 = N1 ^ tag := no,
    show(N2 ^ tag, !IO).

% MERK-LABEL: merk.proc private @field_access.area_f1_0(%arg0: !merk.term<"field_access.shape">) -> i64
% MERK-NEXT: %[[T1:.*]] = merk.has_functor #merk.functor<"rect", 2> %arg0 : !merk.term<"field_access.shape">
% MERK-NEXT: merk.guard %[[T1]]
% MERK-NEXT: %[[W:.*]]:2 = merk.deconstruct #merk.functor<"rect", 2> %arg0 : !merk.term<"field_access.shape"> -> i64, i64
% MERK-NEXT: %[[T2:.*]] = merk.has_functor #merk.functor<"rect", 2> %arg0 : !merk.term<"field_access.shape">
% MERK-NEXT: merk.guard %[[T2]]
% MERK-NEXT: %[[H:.*]]:2 = merk.deconstruct #merk.functor<"rect", 2> %arg0 : !merk.term<"field_access.shape"> -> i64, i64
% MERK-NEXT: arith.muli %[[W]]#0, %[[H]]#1 : i64
% MERK-LABEL: merk.proc private @field_access.put_f2_0(%arg0: !merk.type_info, %arg1: !merk.term<"field_access.box(T0)">, %arg2: !merk.word)
% MERK-NEXT: %[[B:.*]]:2 = merk.deconstruct #merk.functor<"box", 2> %arg1 : !merk.term<"field_access.box(T0)"> -> !merk.word, i64
% MERK-NEXT: %[[N:.*]] = merk.construct #merk.functor<"box", 2>(%arg2, %[[B]]#1 : !merk.word, i64) -> !merk.term<"field_access.box(T0)">
% MERK-NEXT: merk.succeed %[[N]]

% OUTPUT:3
% OUTPUT-NEXT:"p"
% OUTPUT-NEXT:0.5
% OUTPUT-NEXT:point(10, 2, "q", 0.5)
% OUTPUT-NEXT:point(1, 2, "p", 0.5)
% OUTPUT-NEXT:point(10, 20, "q", 2.0)
% OUTPUT-NEXT:12
% OUTPUT-NEXT:"no area"
% OUTPUT-NEXT:2.5
% OUTPUT-NEXT:"no radius"
% OUTPUT-NEXT:rect(1, 7)
% OUTPUT-NEXT:"no"
% OUTPUT-NEXT:box("s", 2)
% OUTPUT-NEXT:box("t", 0)
% OUTPUT-NEXT:[1, 2]
% OUTPUT-NEXT:"p"
% OUTPUT-NEXT:nested(point(99, 2, "p", 0.5), yes(3))
% OUTPUT-NEXT:no
% OUTPUT-EMPTY:
