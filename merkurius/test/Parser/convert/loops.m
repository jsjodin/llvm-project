% Loops, ( Iterators do Body ), as the converter reads them with -fdo-loops
% (design document, "Loops"): each becomes a LoopGoal with its iterators,
% in order. Without -fdo-loops, do is not an operator, as in mmc, and the
% loops are syntax errors.
%
% RUN: merkc -fc1 -fdo-loops -fopenmp -fdebug-dump-parse-tree-no-sema %s \
% RUN:   | FileCheck %s
% RUN: not merkc -fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --check-prefix=PLAIN
% PLAIN: loops.m:{{[0-9]+}}:{{[0-9]+}}: error: expected an operator or ')', but found 'do'
% Semantics/desugar/loops.m desugars them.

:- module loops.
:- interface.
:- import_module array, float, int, list.

:- pred sum_list(list(int)::in, int::in, int::out) is det.

:- implementation.

sum_list(Xs, !Sum) :-
    ( foreach(X, Xs) do !:Sum = !.Sum + X ).
% CHECK:      | | | name: SymName sum_list
% CHECK:      | | | body: LoopGoal
% CHECK-NEXT: | | | | iterators:
% CHECK-NEXT: | | | | | LoopIterator
% CHECK-NEXT: | | | | | | kind: Foreach
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | | | Variable X
% CHECK-NEXT: | | | | | | | Variable Xs
% CHECK-NEXT: | | | | body: Unification

% Every iterator, in lockstep with a for, and a body that is an
% if-then-else, in parentheses.
below_sum(N, Count, Squares) :-
    ( for(I, 1, N), fromto(0, S0, S1, Sum), count(K, 0, _) do
        S1 = S0 + I + K
    ),
    ( for(I, N, 1, -1), fromto(0, C0, C1, Count), foreach(Y, Squares) do
        Y = I * I,
        ( if Y < Sum then C1 = C0 + 1 else C1 = C0 )
    ).
% CHECK:      | | | name: SymName below_sum
% CHECK:      | | | | | LoopGoal
% CHECK-NEXT: | | | | | | iterators:
% CHECK-NEXT: | | | | | | | LoopIterator
% CHECK-NEXT: | | | | | | | | kind: For
% CHECK:      | | | | | | | LoopIterator
% CHECK-NEXT: | | | | | | | | kind: Fromto
% CHECK:      | | | | | | | LoopIterator
% CHECK-NEXT: | | | | | | | | kind: Count
% CHECK:      | | | | | LoopGoal
% CHECK-NEXT: | | | | | | iterators:
% CHECK-NEXT: | | | | | | | LoopIterator
% CHECK-NEXT: | | | | | | | | kind: For
% CHECK-NEXT: | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | | Variable I
% CHECK-NEXT: | | | | | | | | | Variable N
% CHECK-NEXT: | | | | | | | | | Literal 1
% CHECK-NEXT: | | | | | | | | | Literal -1
% CHECK:      | | | | | | | | kind: Fromto
% CHECK:      | | | | | | | | kind: Foreach
% CHECK:      | | | | | | body: Conjunction

% A nest under a directive, which applies to the outer loop, and a
% fromto that a ground end ends.
matmul(N, A, B, !C) :-
    %$omp parallel do collapse(2)
    ( for(I, 0, N - 1) do
        ( for(J, 0, N - 1) do
            ( for(K, 0, N - 1), fromto(0.0, S0, S1, Sum) do
                S1 = S0 + A ^ unsafe_elem(I * N + K) * B ^ unsafe_elem(K * N + J)
            ),
            !C ^ unsafe_elem(I * N + J) := Sum
        )
    ).
% CHECK:      | | | name: SymName matmul
% CHECK:      | | | body: OmpGoal
% CHECK:      | | | | | id: parallel do
% CHECK:      | | | | goal: LoopGoal
% CHECK:      | | | | | body: LoopGoal
% CHECK:      | | | | | | body: Conjunction
% CHECK-NEXT: | | | | | | | parallel: false
% CHECK-NEXT: | | | | | | | goals:
% CHECK-NEXT: | | | | | | | | LoopGoal

traverse(L, Last) :-
    ( fromto(L, [X | Xs], Xs, []), fromto(0, _, X, Last) do true ).
% CHECK:      | | | name: SymName traverse
% CHECK:      | | | body: LoopGoal
% CHECK:      | | | | | | kind: Fromto
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | | | Variable L
% CHECK-NEXT: | | | | | | | ListData
% CHECK:      | | | | body: BuiltinGoal

% do(...) written as a call stays a call, and (do) is an atom.
call_do(X) :-
    do(1, 2),
    X = (do).
% CHECK:      | | | name: SymName call_do
% CHECK:      | | | | | Call
% CHECK-NEXT: | | | | | | name: SymName do
