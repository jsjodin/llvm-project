% The translation of loops (design document, "Loops", Meaning): each loop
% is a call of its loop predicate, which calls its body predicate in each
% iteration, both named after the loop's position, with the clause's types.
% A for's trip count is computed before the loop and counted down; a
% stride other than 1 or -1 computes it by div and mod, so that it cannot
% overflow. Without a for, the loop is a disjunction of Ends and Steps,
% and a fromto whose Last has no variables gives it Last's functor and
% In's pattern, taken from the body. Each goal's nonlocal variables are
% after "~".
%
% RUN: merkc -fc1 -fdo-loops -fdebug-dump-loops %s 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s

:- module translate.
:- interface.
:- import_module list.
:- pred sum_list(list(int)::in, int::in, int::out) is det.
:- pred evens(int::in, list(int)::out) is det.
:- pred down(int::in, int::out) is det.
:- pred length_of(list(T)::in, int::out) is det.
:- pred last(list(int)::in, int::out) is semidet.
:- pred nest(int::in, int::out) is det.
:- implementation.
:- import_module int.

% A threaded state variable: its version after the loop is its version in
% the iteration that does not happen.
sum_list(Xs, !Sum) :-
    ( foreach(X, Xs) do !:Sum = !.Sum + X ).
% CHECK:      [[@LINE-2]]:1 pred translate.sum_list(list.list(int), int, int)
% CHECK-NEXT: Predicate sum_list(Xs, Sum!0, Sum!3)
% CHECK-NEXT: | Call #{{[0-9]+}} sum_list__loop_[[SUMLIST7:[0-9]+]]_7(Xs, Sum!0, Sum!3)
% CHECK-NEXT: {{[0-9]+}}:7 pred translate.sum_list__loop_[[SUMLIST7]]_7(list.list(int), int, int)
% CHECK-NEXT: Predicate sum_list__loop_[[SUMLIST7]]_7($6, Sum!1, Sum!3)
% CHECK-NEXT: | Disj
% CHECK-NEXT: | | Conj
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $6 = []
% CHECK-NEXT: | | | Unify #{{[0-9]+}} Sum!3 = Sum!1
% CHECK-NEXT: | | Conj
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $6 = [X | $7]
% CHECK-NEXT: | | | Call #{{[0-9]+}} sum_list__body_[[SUMLIST7]]_7(X, Sum!1, Sum!2)
% CHECK-NEXT: | | | Call #{{[0-9]+}} sum_list__loop_[[SUMLIST7]]_7($7, Sum!2, Sum!3)
% CHECK-NEXT: {{[0-9]+}}:7 pred translate.sum_list__body_[[SUMLIST7]]_7(int, int, int)
% CHECK-NEXT: Predicate sum_list__body_[[SUMLIST7]]_7(X, Sum!1, Sum!2)
% CHECK-NEXT: | Unify #0 Sum!2 = '+'(Sum!1, X)

% A stride of 2: the trip count by div and mod, and a list built.
evens(N, Ys) :-
    ( for(I, 0, N, 2), foreach(Y, Ys) do Y = I ).
% CHECK:      Predicate evens(N, Ys)
% CHECK:      | | | IfThenElse
% CHECK-NEXT: | | | | Call #{{[0-9]+}} int.'<'(N, $2)
% CHECK-NEXT: | | | | Unify #{{[0-9]+}} $5 = 0
% CHECK-NEXT: | | | | Conj
% CHECK-NEXT: | | | | | Unify #{{[0-9]+}} $7 = 2
% CHECK-NEXT: | | | | | Unify #{{[0-9]+}} $8 = int.div(N, $7)
% CHECK-NEXT: | | | | | Unify #{{[0-9]+}} $9 = int.div($2, $7)
% CHECK-NEXT: | | | | | Unify #{{[0-9]+}} $10 = int.mod(N, $7)
% CHECK-NEXT: | | | | | Unify #{{[0-9]+}} $11 = int.mod($2, $7)
% CHECK-NEXT: | | | | | Unify #{{[0-9]+}} $12 = int.'-'($8, $9)
% CHECK-NEXT: | | | | | IfThenElse
% CHECK-NEXT: | | | | | | Call #{{[0-9]+}} int.'<'($10, $11)
% CHECK-NEXT: | | | | | | Unify #{{[0-9]+}} $5 = $12
% CHECK:      | | | Call #{{[0-9]+}} evens__loop_[[EVENS7:[0-9]+]]_7($2, $5, Ys)
% CHECK:      Predicate evens__loop_[[EVENS7]]_7(I, $6, $18)
% CHECK-NEXT: | IfThenElse
% CHECK-NEXT: | | Unify #{{[0-9]+}} $6 = 0
% CHECK-NEXT: | | Unify #{{[0-9]+}} $18 = []
% CHECK-NEXT: | | Conj
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $18 = [Y | $19]
% CHECK-NEXT: | | | Call #{{[0-9]+}} evens__body_[[EVENS7]]_7(I, Y)
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $14 = 2
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $15 = int.'+'(I, $14)
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $16 = 1
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $17 = int.'-'($6, $16)
% CHECK-NEXT: | | | Call #{{[0-9]+}} evens__loop_[[EVENS7]]_7($15, $17, $19)

% A negative step counts from Lo down to Hi; N, shared, is passed through.
down(N, S) :-
    ( for(I, N, 1, -1), fromto(0, S0, S1, S) do S1 = S0 + I * N ).
% CHECK:      Predicate down(N, S)
% CHECK:      | | | | Call #{{[0-9]+}} int.'<'(N, $2)
% CHECK:      | | | | | Unify #{{[0-9]+}} $10 = int.'-'(N, $2)
% CHECK:      | | | Call #{{[0-9]+}} down__loop_[[DOWN7:[0-9]+]]_7(N, $8, $3, S, N)
% CHECK:      Predicate down__loop_[[DOWN7]]_7(I, $9, S0, S, N)
% CHECK:      | | | Call #{{[0-9]+}} down__body_[[DOWN7]]_7(I, S0, S1, N)
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $12 = -1

% A count's last value is unified with its Hi in Ends; the loop's
% predicates are polymorphic.
length_of(Xs, N) :-
    ( foreach(_, Xs), count(_, 1, N) do true ).
% CHECK:      {{[0-9]+}}:7 pred translate.length_of__loop_[[LENGTHOF7:[0-9]+]]_7(list.list(T), int, int)
% CHECK-NEXT: Predicate length_of__loop_[[LENGTHOF7]]_7($5, _$4, N)
% CHECK-NEXT: | Disj
% CHECK-NEXT: | | Conj
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $5 = []
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $9 = 1
% CHECK-NEXT: | | | Unify #{{[0-9]+}} N = int.'-'(_$4, $9)

% Schimpf's last element: the first fromto's Last, [], and its In's
% pattern, [X | Xs], make the switch.
last(L, Last) :-
    L = [_ | _],
    ( fromto(L, [X | Xs], Xs, []), fromto(0, _, X, Last) do true ).
% CHECK:      Predicate last__loop_[[LAST7:[0-9]+]]_7($6, _$9, Last)
% CHECK-NEXT: | Disj
% CHECK-NEXT: | | Conj
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $6 = []
% CHECK-NEXT: | | | Unify #{{[0-9]+}} Last = _$9
% CHECK-NEXT: | | Conj
% CHECK-NEXT: | | | Unify #{{[0-9]+}} $6 = [X | Xs]
% CHECK-NEXT: | | | Call #{{[0-9]+}} last__body_[[LAST7]]_7
% CHECK-NEXT: | | | Call #{{[0-9]+}} last__loop_[[LAST7]]_7(Xs, X, Last)

% A nest: the inner loop is in the outer body predicate, which computes
% its trip count, and shares the outer counter.
nest(N, Total) :-
    ( for(I, 1, N), fromto(0, T0, T1, Total) do
        ( for(J, 1, I), fromto(T0, U0, U1, T1) do U1 = U0 + I * J )
    ).
% CHECK:      Predicate nest__body_[[NEST7:[0-9]+]]_7(I, T0, T1)
% CHECK:      | | | Call #{{[0-9]+}} nest__loop_[[NEST11:[0-9]+]]_11($7, $12, T0, T1, I)
% CHECK:      Predicate nest__loop_[[NEST11]]_11(J, $13, U0, T1, I)
% CHECK:      Predicate nest__body_[[NEST11]]_11(J, U0, U1, I)
% CHECK-NEXT: | Conj
% CHECK-NEXT: | | Unify #{{[0-9]+}} U1 = '+'(U0, $11)
% CHECK-NEXT: | | Unify #{{[0-9]+}} $11 = '*'(I, J)
