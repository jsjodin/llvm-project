% Loops are typed in their clause, as lambda expressions are (design
% document, "Loops"): a counter and its bounds are ints, a foreach's list
% is a list of its element's type, a fromto's arguments have one type, and
% so do a threaded state variable's versions. Their types are not
% generalized on their own, but a predicate whose types are inferred may
% have a loop, and is generalized as usual. Inputs/loops-translation.m is
% the translation of each loop into recursive predicates, written by hand
% with the types below declared, which loops-translation.test checks with
% mmc. The loops' predicates, which their translation makes, have the
% types given here, and their modes are inferred.
%
% RUN: merkc -fc1 -fdo-loops -fdebug-dump-types %s 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: merkc -fc1 -fdo-loops -fdebug-dump-modes %s 2>&1 \
% RUN:   | FileCheck --check-prefix=MODES --implicit-check-not=error: %s
% MODES: 37:7 pred loops.sum_list__loop_37_7/3 inferred mode 0: sum_list__loop_37_7(in, in, out)
% MODES: 83:7 pred loops.saxpy__loop_83_7/6 inferred mode 0: saxpy__loop_83_7(in, di, di, uo, in, in)
% MODES: 114:7 pred loops.length_of__loop_114_7/3 inferred mode 0: length_of__loop_114_7(in, di, uo)

:- module loops.
:- interface.
:- import_module array, list, pair.

:- pred sum_list(list(int)::in, int::in, int::out) is det.
:- pred pi(int::in, float::out) is det.
:- pred sum_floats(list(float)::in, float::out) is det.
:- pred keys(list(pair(string, int))::in, list(string)::out, int::out) is det.
:- pred saxpy(float::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
:- pred triangle(int::in, int::out) is det.
:- pred inferred(list(int)::in, int::out, int::out) is det.

:- implementation.
:- import_module float, int.

sum_list(Xs, !Sum) :-
    ( foreach(X, Xs) do !:Sum = !.Sum + X ).
% CHECK:      clause at [[@LINE-2]]:1 of pred loops.sum_list/3: pred loops.sum_list(list.list(int), int, int)
% CHECK-NEXT:   Sum!0: int
% CHECK-NEXT:   Xs: list.list(int)
% CHECK-NEXT:   Sum!1: int
% CHECK-NEXT:   X: int
% CHECK-NEXT:   Sum!2: int
% CHECK-NEXT:   Sum!3: int
% CHECK-NEXT:   [[@LINE-8]]:39 + -> func int.'+'/2

% The fromto's First, 0.0, makes the accumulator a float, which chooses
% float's + and /.
pi(N, Pi) :-
    Step = 1.0 / float(N),
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        X = (float(I) + 0.5) * Step,
        Y = 4.0 / (1.0 + X * X),
        S1 = S0 + Y
    ),
    Pi = Step * Sum.
% CHECK:      clause at [[@LINE-8]]:1 of pred loops.pi/2: pred loops.pi(int, float)
% CHECK:        I: int
% CHECK-NEXT:   S0: float
% CHECK-NEXT:   S1: float
% CHECK-NEXT:   X: float
% CHECK-NEXT:   Y: float
% CHECK:        [[@LINE-9]]:17 + -> func float.'+'/2

% The list's element type chooses +.
sum_floats(Xs, Sum) :-
    ( foreach(X, Xs), fromto(0.0, S0, S1, Sum) do S1 = S0 + X ).
% CHECK:      clause at [[@LINE-2]]:1 of pred loops.sum_floats/2: pred loops.sum_floats(list.list(float), float)
% CHECK:        [[@LINE-2]]:59 + -> func float.'+'/2

% A pattern for foreach's element, and lists built and walked in lockstep.
keys(Pairs, Keys, Total) :-
    ( foreach(K - V, Pairs), foreach(K, Keys), fromto(0, T0, T1, Total) do
        T1 = T0 + V
    ).
% CHECK:      clause at [[@LINE-4]]:1 of pred loops.keys/3: pred loops.keys(list.list(pair.pair(string, int)), list.list(string), int)
% CHECK:        K: string
% CHECK-NEXT:   V: int

% An array threaded through a loop, updated in place.
saxpy(A, X, !Y) :-
    N = array.size(X),
    ( for(I, 0, N - 1) do
        !Y ^ unsafe_elem(I) := A * X ^ unsafe_elem(I) + !.Y ^ unsafe_elem(I)
    ).
% CHECK:      clause at [[@LINE-5]]:1 of pred loops.saxpy/4: pred loops.saxpy(float, array.array(float), array.array(float), array.array(float))
% CHECK:        Y!1: array.array(float)
% CHECK-NEXT:   I: int
% CHECK-NEXT:   Y!2: array.array(float)

% A nest, whose inner loop's bound is the outer loop's counter.
triangle(N, Total) :-
    ( for(I, 1, N), fromto(0, T0, T1, Total) do
        ( for(J, 1, I), fromto(T0, U0, U1, T1) do U1 = U0 + I * J )
    ).
% CHECK:      clause at [[@LINE-4]]:1 of pred loops.triangle/2: pred loops.triangle(int, int)
% CHECK:        J: int
% CHECK-NEXT:   U0: int
% CHECK-NEXT:   U1: int

% Predicates whose types are inferred, with loops in them: count_positive
% is for ints, and length_of is polymorphic.
inferred(Xs, Positive, Length) :-
    count_positive(Xs, Positive),
    length_of(Xs, Length).

count_positive(Xs, N) :-
    ( foreach(X, Xs), fromto(0, C0, C1, N) do
        ( if X > 0 then C1 = C0 + 1 else C1 = C0 )
    ).
% CHECK:      clause at [[@LINE-4]]:1 of pred loops.count_positive/2: inferred pred loops.count_positive(list.list(int), int)

length_of(Xs, N) :-
    ( foreach(_, Xs), count(_, 1, N) do true ).
% CHECK:      clause at [[@LINE-2]]:1 of pred loops.length_of/2: inferred pred loops.length_of(list.list(T1), int)
