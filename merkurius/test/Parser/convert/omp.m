% %$omp directives, as the converter reads them with -fopenmp: each one
% applies to the goal after it, which becomes an OmpGoal. Without -fopenmp
% they are comments, so the module is the sequential program that mmc
% compiles.
%
% RUN: merkc -fc1 -fopenmp -fdebug-dump-parse-tree-no-sema %s | FileCheck %s
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck %s --check-prefix=SEQ
% SEQ-NOT: OmpGoal

:- module omp.
:- interface.
:- import_module array, int.

:- pred sum_squares(array(int)::in, int::in, int::out) is det.
:- pred scale(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pred count_up(int::in, int::in, int::in, int::out) is det.
:- pred count_up_in(array(int)::in, int::in, int::in, int::in, int::out)
    is det.

:- implementation.

% The running example of the design document.
sum_squares(A, N, Sum) :-
    %$omp parallel do reduction(+)
    int.fold_up(add_square(A), 0, N - 1, 0, Sum).
% CHECK:      | | | body: OmpGoal
% CHECK-NEXT: | | | | directive: OmpDirective
% CHECK-NEXT: | | | | | id: parallel do
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | | | clauses:
% CHECK-NEXT: | | | | | | OmpReductionClause
% CHECK-NEXT: | | | | | | | modifier: none
% CHECK-NEXT: | | | | | | | op: SymName +
% CHECK-NEXT: | | | | | | | variables:
% CHECK-NEXT: | | | | goal: Call
% CHECK-NEXT: | | | | | name: SymName int.fold_up

:- pred add_square(array(int)::in, int::in, int::in, int::out) is det.

add_square(A, I, !Acc) :-
    array.unsafe_lookup(A, I, X),
    !:Acc = !.Acc + X * X.

% A schedule with a modifier and a chunk size, clauses separated by commas
% and continued on a second line, and clauses that keep their arguments as
% data-terms.
scale(K, !A) :-
    Size = array.size(!.A),
    %$omp parallel do schedule(monotonic: dynamic, Size / 8),
    %$omp&    num_threads(4), private(K) collapse(1) if(Size > 1000)
    int.fold_up(scale_element(K), 0, Size - 1, !A),
    scale_element(K, 0, !A).
% CHECK:      | | | | | OmpGoal
% CHECK-NEXT: | | | | | | directive: OmpDirective
% CHECK-NEXT: | | | | | | | id: parallel do
% CHECK-NEXT: | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | clauses:
% CHECK-NEXT: | | | | | | | | OmpScheduleClause
% CHECK-NEXT: | | | | | | | | | modifiers:
% CHECK-NEXT: | | | | | | | | | | Monotonic
% CHECK-NEXT: | | | | | | | | | kind: Dynamic
% CHECK-NEXT: | | | | | | | | | chunk: FunctorData
% CHECK-NEXT: | | | | | | | | | | name: SymName /
% CHECK-NEXT: | | | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | | | | Variable Size
% CHECK-NEXT: | | | | | | | | | | | Literal 8
% CHECK-NEXT: | | | | | | | | OmpGenericClause
% CHECK-NEXT: | | | | | | | | | id: num_threads
% CHECK-NEXT: | | | | | | | | | modifiers:
% CHECK-NEXT: | | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | | | Literal 4
% CHECK-NEXT: | | | | | | | | OmpGenericClause
% CHECK-NEXT: | | | | | | | | | id: private
% CHECK-NEXT: | | | | | | | | | modifiers:
% CHECK-NEXT: | | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | | | Variable K
% CHECK-NEXT: | | | | | | | | OmpGenericClause
% CHECK-NEXT: | | | | | | | | | id: collapse
% CHECK:      | | | | | | | | OmpGenericClause
% CHECK-NEXT: | | | | | | | | | id: if
% CHECK-NEXT: | | | | | | | | | modifiers:
% CHECK-NEXT: | | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | | | FunctorData
% CHECK-NEXT: | | | | | | | | | | | name: SymName >
% CHECK:      | | | | | | goal: Call
% CHECK-NEXT: | | | | | | | name: SymName int.fold_up
% A directive applies to one conjunct.
% CHECK-NEXT: | | | | | | | arguments:
% CHECK:      | | | | | Call
% CHECK-NEXT: | | | | | | name: SymName scale_element

:- pred scale_element(int::in, int::in,
    array(int)::array_di, array(int)::array_uo) is det.

scale_element(K, I, !A) :-
    array.unsafe_lookup(!.A, I, X),
    array.unsafe_set(I, K * X, !A).

% A self-recursive counted loop, with the Merkurius clauses index and
% upper, and the combined directive that a target region would use, which
% maps a section of an array.
count_up(I, N, !Acc) :-
    count_up_in(array.make_empty_array, I, N, !Acc).

count_up_in(A, I, N, !Acc) :-
    ( if I =< N then
        %$omp target teams distribute parallel do simd index(I) upper(N)
        %$omp&    map(always, to: N) map(tofrom: Acc0, A(I .. N - 1))
        %$omp&    reduction(task, max: Acc0)
        count_up_in(A, I + 1, N, !Acc),
        Acc0 = !.Acc,
        !:Acc = int.max(Acc0, I)
    else
        true
    ).
% CHECK:      | | | | | | OmpGoal
% CHECK-NEXT: | | | | | | | directive: OmpDirective
% CHECK-NEXT: | | | | | | | | id: target teams distribute parallel do simd
% CHECK-NEXT: | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | clauses:
% CHECK-NEXT: | | | | | | | | | OmpIndexClause
% CHECK-NEXT: | | | | | | | | | | variable: Variable I
% CHECK-NEXT: | | | | | | | | | OmpUpperClause
% CHECK-NEXT: | | | | | | | | | | bound: Variable N
% CHECK-NEXT: | | | | | | | | | OmpMapClause
% CHECK-NEXT: | | | | | | | | | | modifiers:
% CHECK-NEXT: | | | | | | | | | | | Always
% CHECK-NEXT: | | | | | | | | | | type: To
% CHECK-NEXT: | | | | | | | | | | objects:
% CHECK-NEXT: | | | | | | | | | | | OmpMapObject
% CHECK-NEXT: | | | | | | | | | | | | variable: Variable N
% CHECK-NEXT: | | | | | | | | | | | | lower: none
% CHECK-NEXT: | | | | | | | | | | | | upper: none
% CHECK-NEXT: | | | | | | | | | OmpMapClause
% CHECK-NEXT: | | | | | | | | | | modifiers:
% CHECK-NEXT: | | | | | | | | | | type: Tofrom
% CHECK-NEXT: | | | | | | | | | | objects:
% CHECK-NEXT: | | | | | | | | | | | OmpMapObject
% CHECK-NEXT: | | | | | | | | | | | | variable: Variable Acc0
% CHECK-NEXT: | | | | | | | | | | | | lower: none
% CHECK-NEXT: | | | | | | | | | | | | upper: none
% An array section, with inclusive bounds.
% CHECK-NEXT: | | | | | | | | | | | OmpMapObject
% CHECK-NEXT: | | | | | | | | | | | | variable: Variable A
% CHECK-NEXT: | | | | | | | | | | | | lower: Variable I
% CHECK-NEXT: | | | | | | | | | | | | upper: FunctorData
% CHECK-NEXT: | | | | | | | | | | | | | name: SymName -
% CHECK-NEXT: | | | | | | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | | | | | | | Variable N
% CHECK-NEXT: | | | | | | | | | | | | | | Literal 1
% CHECK-NEXT: | | | | | | | | | OmpReductionClause
% CHECK-NEXT: | | | | | | | | | | modifier: Task
% CHECK-NEXT: | | | | | | | | | | op: SymName max
% CHECK-NEXT: | | | | | | | | | | variables:
% CHECK-NEXT: | | | | | | | | | | | Variable Acc0
% CHECK-NEXT: | | | | | | | goal: Call
% CHECK-NEXT: | | | | | | | | name: SymName count_up_in

% A directive with arguments, before the goal of an else branch.
:- pred locked(int::in, int::out) is det.

locked(X, Y) :-
    ( if X > 0 then
        Y = X
    else
        %$omp critical(counter) hint(2)
        Y = 0
    ).
% CHECK:      | | | | else: OmpGoal
% CHECK-NEXT: | | | | | directive: OmpDirective
% CHECK-NEXT: | | | | | | id: critical
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | | | FunctorData
% CHECK-NEXT: | | | | | | | | name: SymName counter
% CHECK-NEXT: | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | clauses:
% CHECK-NEXT: | | | | | | | OmpGenericClause
% CHECK-NEXT: | | | | | | | | id: hint
% CHECK-NEXT: | | | | | | | | modifiers:
% CHECK-NEXT: | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | | Literal 2
% CHECK-NEXT: | | | | | goal: Unification
