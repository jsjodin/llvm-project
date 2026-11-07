% Conjunctions of unifications and calls that mode analysis accepts, and
% how it runs them (Overton, "Precise and Expressive Mode Systems for
% Typed Logic Programming Languages", 2003, chapter 3): goals in source
% order, but each that needs a variable the goals after it bind waiting
% for them; what each unification does; the procedure each call runs,
% including a function's reverse mode and an implied mode, whose bound
% argument the call outputs to a fresh variable to test; and the head's
% insts at the end. A procedure with what is not checked yet says so.
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module conjunctions.
:- interface.
:- import_module list.

:- type r ---> r(a :: int, b :: int).

:- pred reorder(int::out) is det.
:- pred unifications(list(int)::in, int::out, list(int)::out) is semidet.
:- pred selection(int::out) is det.
:- pred reverse(int::in, int::out) is det.
:- pred implied(int::in) is semidet.
:- pred fields(r::in, r::out, int::out) is det.
:- pred unique(list(int)::uo) is det.
:- pred pass(T::di, T::uo) is det.
:- func double(int) = int.
:- pred later(int::out) is det.
:- pred parallel(int::out, int::out) is det.

:- implementation.
:- import_module int.

:- pred w(int::in, int::out) is det.
w(X, X).
:- pred m(int, int).
:- mode m(in, out) is det.
:- mode m(out, in) is det.
m(X, X).

reorder(X) :-
    w(Y, X),
    Y = Z,
    Z = 3.
% CHECK-LABEL: pred conjunctions.reorder/1 mode 0: reorder(out) is det
% CHECK-NEXT: clause at [[@LINE-5]]:1
% CHECK-NEXT: [[@LINE-3]]:{{[0-9]+}} construct Z = 3
% CHECK-NEXT: [[@LINE-5]]:{{[0-9]+}} assign Y := Z
% CHECK-NEXT: [[@LINE-7]]:{{[0-9]+}} call pred conjunctions.w/2 mode 0: w(Y, X)
% CHECK-NEXT: final: X ground

unifications(L, H, T) :-
    L = [H | T0],
    T0 = T,
    H = 1,
    T = [_ | _].
% CHECK-LABEL: pred conjunctions.unifications/3 mode 0: unifications(in, out, out) is semidet
% CHECK-NEXT: clause at [[@LINE-6]]:1
% CHECK-NEXT: [[@LINE-6]]:{{[0-9]+}} deconstruct L = '[|]'(H, T0)
% CHECK-NEXT: [[@LINE-6]]:{{[0-9]+}} assign T := T0
% CHECK-NEXT: [[@LINE-6]]:{{[0-9]+}} test H = 1
% CHECK-NEXT: [[@LINE-6]]:{{[0-9]+}} deconstruct T = '[|]'(V_{{[0-9]+}}, V_{{[0-9]+}})
% CHECK-NEXT: final: L bound('[|]'(ground, ground)), H bound(1), T bound('[|]'(ground, ground))

selection(X) :-
    m(X, 3).
% CHECK-LABEL: pred conjunctions.selection/1 mode 0: selection(out) is det
% CHECK: call pred conjunctions.m/2 mode 1: m(X, V_{{[0-9]+}})

reverse(X, Y) :-
    X = Y + 1.
% CHECK-LABEL: pred conjunctions.reverse/2 mode 0: reverse(in, out) is det
% CHECK: call func int.'+'/2 mode 1: X = +(Y, V_{{[0-9]+}})
% CHECK-NEXT: final: X ground, Y unique

implied(X) :-
    w(X, 3).
% CHECK-LABEL: pred conjunctions.implied/1 mode 0: implied(in) is semidet
% CHECK: call pred conjunctions.w/2 mode 0 implied: w(X, V_{{[0-9]+}})

fields(R0, R, A) :-
    A = R0 ^ a,
    R = R0 ^ b := A.
% CHECK-LABEL: pred conjunctions.fields/3 mode 0: fields(in, out, out) is det
% CHECK-NEXT: clause at [[@LINE-4]]:1
% CHECK-NEXT: [[@LINE-4]]:{{[0-9]+}} call field conjunctions.a/1 mode 0: A = a(R0)
% CHECK-NEXT: [[@LINE-4]]:{{[0-9]+}} call field update conjunctions.'b :='/2 mode 0: R = b :=(R0, A)
% CHECK-NEXT: final: R0 ground, R ground, A ground

unique(L) :-
    L = [1, 2].
% CHECK-LABEL: pred conjunctions.unique/1 mode 0: unique(uo) is det
% CHECK: final: L unique('[|]'(unique(1), unique('[|]'(unique(2), unique([])))))

pass(X, X).
% CHECK-LABEL: pred conjunctions.pass/2 mode 0: pass(di, uo) is det
% CHECK: assign HeadVar__2 := X
% CHECK-NEXT: final: X unique, HeadVar__2 unique

double(X) = X * 2.
% CHECK-LABEL: func conjunctions.double/1 mode 0: double(in) = out is det
% CHECK: call func int.'*'/2 mode 0: HeadVar__2 = *(X, V_{{[0-9]+}})

later(X) :-
    trace [compile_time(flag("later"))] true,
    X = 1.
% CHECK-LABEL: pred conjunctions.later/1 mode 0: later(out) is det
% CHECK-NEXT: clause at
% CHECK-NEXT: construct X = 1

parallel(X, Y) :-
    ( X = 1 & Y = 2 ).
% CHECK-LABEL: pred conjunctions.parallel/2 mode 0: parallel(out, out) is det
% CHECK-NEXT: not checked: ParallelConjunction
