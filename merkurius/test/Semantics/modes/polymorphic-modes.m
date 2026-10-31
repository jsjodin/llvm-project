% Constrained polymorphic modes that mode analysis accepts (reference
% manual, "Constrained polymorphic modes"; Overton, "Precise and
% Expressive Mode Systems for Typed Logic Programming Languages", 2003,
% section 4.2): a clause whose head has an inst variable, which a value
% keeps while a unification only narrows it, and which a merge keeps if
% every branch has it; a call that binds a callee's inst variable to an
% argument's inst, ground or bound, so that its outputs get that inst,
% the manual's append with list_skel; a unification of two values with
% inst variables, which has both; a higher-order inst kept through an
% inst variable, the use the manual names; and a lambda expression and a
% higher-order term with inst variables in their modes.
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module polymorphic_modes.
:- interface.
:- import_module list.

:- type t ---> a ; b ; c ; f(int).

:- pred app(list(T), list(T), list(T)).
:- mode app(in(list_skel(I)), in(list_skel(I)), out(list_skel(I))) is det.
:- pred narrow(t::in(I), t::out(I)) is semidet.
:- pred merged(t::in(I), t::out(I)) is nondet.
:- pred bound_insts(list(t)::out(list_skel(bound(a ; b)))) is multi.
:- pred ground_list(list(t)::in, list(t)::out) is det.
:- pred two(t::in(I), t::in(J), t::out(J)) is semidet.
:- pred keep(int::out) is det.
:- pred lambda(t::in(I), t::out(I)) is det.
:- pred name(t::in(I), t::out(I)) is det.

:- implementation.

:- pred id(T::in(I), T::out(I)) is det.
id(X, X).

app([], Ys, Ys).
app([X | Xs], Ys, [X | Zs]) :-
    app(Xs, Ys, Zs).
% CHECK-LABEL: pred polymorphic_modes.app/3 mode 0: app(in(list.list_skel((I =< ground))), in(list.list_skel((I =< ground))), out(list.list_skel((I =< ground)))) is det
% CHECK: final: HeadVar__1 list.list_skel((I =< ground)), HeadVar__2 list.list_skel((I =< ground)), HeadVar__3 list.list_skel((I =< ground))
% CHECK: call pred polymorphic_modes.app/3 mode 0: app(Xs, Ys, Zs)
% CHECK: final: HeadVar__1 list.list_skel((I =< ground)), HeadVar__2 list.list_skel((I =< ground)), HeadVar__3 unique('[|]'((I =< ground), list.list_skel((I =< ground))))

narrow(X, Y) :-
    X = a,
    Y = X.
% CHECK-LABEL: pred polymorphic_modes.narrow/2 mode 0: narrow(in((I =< ground)), out((I =< ground))) is semidet
% CHECK: final: X (I =< bound(a)), Y (I =< bound(a))

merged(X, Y) :-
    (
        X = a,
        Y = X
    ;
        Y = X
    ).
% CHECK-LABEL: pred polymorphic_modes.merged/2 mode 0: merged(in((I =< ground)), out((I =< ground))) is nondet
% CHECK: final: X (I =< ground), Y (I =< ground)

bound_insts(L) :-
    A = a,
    B = b,
    ( C = A ; C = B ),
    app([C], [A], L).
% CHECK-LABEL: pred polymorphic_modes.bound_insts/1 mode 0: bound_insts(out(list.list_skel(bound(a ; b)))) is multi
% CHECK: call pred polymorphic_modes.app/3 mode 0: app(V_{{[0-9]+}}, V_{{[0-9]+}}, L)
% CHECK-NEXT: final: L list.list_skel(bound(a ; b))

ground_list(L0, L) :-
    app(L0, [], L).
% CHECK-LABEL: pred polymorphic_modes.ground_list/2 mode 0: ground_list(in, out) is det
% CHECK: final: L0 {{.*}}, L list.list_skel(ground)

two(X, Y, Z) :-
    X = Y,
    Z = X.
% CHECK-LABEL: pred polymorphic_modes.two/3 mode 0: two(in((I =< ground)), in((J =< ground)), out((J =< ground))) is semidet
% CHECK: final: X (I, J =< ground), Y (I, J =< ground), Z (I, J =< ground)

keep(Y) :-
    P = (pred(X::out) is det :- X = 1),
    id(P, Q),
    Q(Y).
% CHECK-LABEL: pred polymorphic_modes.keep/1 mode 0: keep(out) is det
% CHECK: call pred polymorphic_modes.id/2 mode 0: id(P, Q)
% CHECK-NEXT: higher-order call: Q(Y)

lambda(X, Y) :-
    P = (pred(U::in(J), V::out(J)) is det :- V = U),
    P(X, Y).
% CHECK-LABEL: pred polymorphic_modes.lambda/2 mode 0: lambda(in((I =< ground)), out((I =< ground))) is det
% CHECK: final: X (I =< ground), Y (I =< ground)

name(X, Y) :-
    P = id,
    P(X, Y).
% CHECK-LABEL: pred polymorphic_modes.name/2 mode 0: name(in((I =< ground)), out((I =< ground))) is det
% CHECK: construct P = id, closure of pred polymorphic_modes.id/2 mode 0
% CHECK: final: X (I =< ground), Y (I =< ground)
