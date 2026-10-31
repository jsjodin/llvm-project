% The procedures of predicates and functions: from mode declarations, from
% declarations that combine types and modes, from `with_inst`, from type
% class methods, and the default mode of a function without mode
% declarations (reference manual, "Predicate and function mode
% declarations"). Named insts and modes are expanded one level.
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module procedures.
:- interface.
:- import_module list.

:- type abc ---> a ; b ; c ; d.
:- type rec ---> nil ; cons(int, rec).
:- inst abc ---> a ; b ; c.
:- inst ilist(I) == list_skel(I).
:- inst rec ---> nil ; cons(ground, rec).
:- mode inabc == in(abc).

:- typeclass c(T) where [
    pred m(T, T),
    mode m(in, out) is det,
    mode m(out, in) is det,
    func fm(T) = T,
    pred n(T::in) is semidet
].

:- pred p(abc::inabc, abc::out) is det.
:- pred q(int, int).
:- mode q(in, out) is det.
:- mode q(out, in) is det.
:- func f(int) = int.
:- func g(int) = int.
:- mode g(in) = out is semidet.
:- pred h(list(int)::in(ilist(ground)), list(int)::out(list_skel(free))) is det.
:- pred r(rec::in(rec)) is det.
:- pred u(T::di, T::uo) is det.
:- pred poly(T::in(I), T::out(I)) is det <= I =< ground.
:- type ho == (pred(int, int)).
:- inst ho == (pred(in, out) is det).
:- pred w(int::in) `with_type` ho `with_inst` ho.
:- pred z is det.

:- implementation.
:- import_module int.

p(X, X).
q(X, X).
f(X) = X.
g(X) = X :- X > 0.
h(_, []).
r(_).
u(X, X).
poly(X, X).
w(_, X, X).
z.

% CHECK: 16:9 inst procedures.abc/0 == bound(a ; b ; c)
% CHECK-NEXT: 17:9 inst procedures.ilist/1 == list.list_skel(I)
% CHECK-NEXT: 18:9 inst procedures.rec/0 == bound(cons(ground, procedures.rec) ; nil)
% CHECK-NEXT: 41:9 inst procedures.ho/0 == (pred(in, out) is det)
% CHECK-NEXT: 23:10 pred procedures.m/2 mode 0: m(in, out) is det
% CHECK-NEXT: 24:10 pred procedures.m/2 mode 1: m(out, in) is det
% CHECK-NEXT: (default) func procedures.fm/1 mode 0: fm(in) = out is det
% CHECK-NEXT: 26:10 pred procedures.n/1 mode 0: n(in) is semidet
% CHECK-NEXT: 29:9 pred procedures.p/2 mode 0: p(in(procedures.abc), out) is det
% CHECK-NEXT: 31:9 pred procedures.q/2 mode 0: q(in, out) is det
% CHECK-NEXT: 32:9 pred procedures.q/2 mode 1: q(out, in) is det
% CHECK-NEXT: (default) func procedures.f/1 mode 0: f(in) = out is det
% CHECK-NEXT: 35:9 func procedures.g/1 mode 0: g(in) = out is semidet
% CHECK-NEXT: 36:9 pred procedures.h/2 mode 0: h(in(procedures.ilist(ground)), out(list.list_skel(free))) is det
% CHECK-NEXT: 37:9 pred procedures.r/1 mode 0: r(in(procedures.rec)) is det
% CHECK-NEXT: 38:9 pred procedures.u/2 mode 0: u(di, uo) is det
% CHECK-NEXT: 39:9 pred procedures.poly/2 mode 0: poly(in(I), out(I)) is det, I =< ground
% CHECK-NEXT: 43:9 pred procedures.z/0 mode 0: z is det
% CHECK-NEXT: 42:9 pred procedures.w/3 mode 0: w(in, in, out) is det
