% Higher-order terms and calls that mode analysis accepts, and how it
% runs them (reference manual, "Higher-order"; Overton, "Precise and
% Expressive Mode Systems for Typed Logic Programming Languages", 2003,
% 3.2.5): a lambda expression, which waits for its nonlocals, and whose
% body is checked as a clause's is; a function's lambda expression
% without modes; a predicate's or function's name, with or without its
% first arguments, including one of several modes chosen by its
% higher-order arguments; calls of each, with implied modes, by its
% higher-order inst, or for a function without one by the default; a
% higher-order inst through an output, a list and branches; and a
% library predicate with a higher-order argument.
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module higher_order.
:- interface.
:- import_module list.

:- pred lambda(int::in, int::out) is det.
:- pred function(int::out) is det.
:- pred names(int::out) is semidet.
:- pred chosen(int::out) is det.
:- pred implied(int::out) is semidet.
:- pred default((func(int) = int)::in, int::out) is det.
:- pred output(pred(int)::out(pred(out) is det)) is det.
:- pred through(int::in, int::out) is det.
:- pred library(list(int)::in, list(int)::out) is det.

:- implementation.
:- import_module int.

:- pred w(int::in, int::out) is det.
w(X, X).
:- pred t is semidet.
t :- semidet_true.
:- func plus1(int, int) = int.
plus1(X, Y) = X + Y.
:- pred app(pred(int), int).
:- mode app(in(pred(in) is det), in) is det.
:- mode app(in(pred(out) is det), out) is det.
app(P, X) :- P(X).
:- pred d(int::in) is det.
d(_).
:- pred o(int::out) is det.
o(1).

lambda(A, Y) :-
    P = (pred(X::out) is det :-
        X = B + 1
    ),
    B = A,
    P(Y).
% CHECK-LABEL: pred higher_order.lambda/2 mode 0: lambda(in, out) is det
% CHECK-NEXT: clause at [[@LINE-7]]:1
% CHECK-NEXT: [[@LINE-4]]:5 assign B := A
% CHECK-NEXT: [[@LINE-8]]:10 construct P = lambda expression
% CHECK-NEXT: [[@LINE-8]]:17 construct V_{{[0-9]+}} = 1
% CHECK-NEXT: [[@LINE-9]]:13 call func int.'+'/2 mode 0: X = +(B, V_{{[0-9]+}})
% CHECK-NEXT: [[@LINE-7]]:5 higher-order call: P(Y)
% CHECK-NEXT: final: A ground, Y ground

function(Y) :-
    F = (func(X) = X * 2),
    Y = apply(F, 3).
% CHECK-LABEL: pred higher_order.function/1 mode 0: function(out) is det
% CHECK: construct F = lambda expression
% CHECK: higher-order call: Y = F(V_{{[0-9]+}})

names(Y) :-
    P = t,
    call(P),
    Q = w,
    Q(1, Z),
    F = plus1(Z),
    Y = F(2).
% CHECK-LABEL: pred higher_order.names/1 mode 0: names(out) is semidet
% CHECK: construct P = t, closure of pred higher_order.t/0 mode 0
% CHECK: higher-order call: P()
% CHECK: construct Q = w, closure of pred higher_order.w/2 mode 0
% CHECK: higher-order call: Q(V_{{[0-9]+}}, Z)
% CHECK: construct F = plus1(Z), closure of func higher_order.plus1/2 mode 0
% CHECK: higher-order call: Y = F(V_{{[0-9]+}})

chosen(Y) :-
    P = app(d),
    P(5),
    Q = app(o),
    Q(Y).
% CHECK-LABEL: pred higher_order.chosen/1 mode 0: chosen(out) is det
% CHECK: construct P = app(V_{{[0-9]+}}), closure of pred higher_order.app/2 mode 0
% CHECK: construct Q = app(V_{{[0-9]+}}), closure of pred higher_order.app/2 mode 1

implied(Y) :-
    Q = w(1),
    Q(1),
    Y = 1.
% CHECK-LABEL: pred higher_order.implied/1 mode 0: implied(out) is semidet
% CHECK: higher-order call implied: Q(V_{{[0-9]+}})

default(F, Y) :-
    Y = F(1).
% CHECK-LABEL: pred higher_order.default/2 mode 0: default(in, out) is det
% CHECK: higher-order call: Y = F(V_{{[0-9]+}})
% CHECK-NEXT: final: F ground, Y ground

output(P) :-
    P = w(1).
% CHECK-LABEL: pred higher_order.output/1 mode 0: output(out((pred(out) is det))) is det
% CHECK: final: P /* unique */ (pred(out) is det)

through(A, Y) :-
    ( if A = 1 then
        P = w(1)
    else
        P = w(2)
    ),
    L = [P],
    L = [Q | _],
    Q(Y).
% CHECK-LABEL: pred higher_order.through/2 mode 0: through(in, out) is det
% CHECK: deconstruct L = '[|]'(Q, V_{{[0-9]+}})
% CHECK-NEXT: higher-order call: Q(Y)

library(L0, L) :-
    list.map((pred(X::in, Y::out) is det :- Y = X + 1), L0, L).
% CHECK-LABEL: pred higher_order.library/2 mode 0: library(in, out) is det
% CHECK: call pred list.map/3 mode 0: map(V_{{[0-9]+}}, L0, L)
