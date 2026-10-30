% Errors in %$omp directives, and directives that apply to no goal, which
% are ignored with a warning. An item with an error is left out. Without
% -fopenmp the directives are comments, and mmc compiles the module.
%
% RUN: not merkc -fc1 -fopenmp -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --check-prefix=SEQ
% SEQ-NOT: error:
% SEQ-NOT: warning:

:- module omp_errors.
:- interface.
:- import_module list.

:- pred c_spelling is det.
:- pred end_directive is det.
:- pred unknown_directive is det.
:- pred unknown_clause is det.
:- pred schedule_kind is det.
:- pred schedule_arguments is det.
:- pred trailing_comma is det.
:- pred leading_comma is det.
:- pred index_variable(int::in) is det.
:- pred map_modifier(int::in) is det.
:- pred map_object(int::in) is det.
:- pred unclosed is det.
:- pred empty is det.
:- pred reduction_list(int::in) is det.
:- pred before_head is det.
:- pred in_dcg(list(int)::in, list(int)::out) is det.
:- pred in_data_term(int::out) is det.

:- implementation.

c_spelling :-
    %$omp parallel for
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:11: error: 'parallel for' is C's form of the OpenMP directive; Mercury uses Fortran's, 'parallel do'

end_directive :-
    %$omp end parallel
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:11: error: 'end' is not an OpenMP directive

unknown_directive :-
    %$omp parallelise
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:11: error: 'parallelise' is not an OpenMP directive

unknown_clause :-
    %$omp parallel frobnicate(3)
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:20: error: 'frobnicate' is not an OpenMP clause

schedule_kind :-
    %$omp do schedule(sometimes)
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:23: error: expected a schedule kind: static, dynamic, guided, auto or runtime, but found 'sometimes'

schedule_arguments :-
    %$omp do schedule(static, 1, 2)
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:14: error: expected schedule([Modifiers :] Kind [, Chunk])

trailing_comma :-
    %$omp do reduction(+),
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:27: error: expected an OpenMP clause, but found the end of the directive

leading_comma :-
    %$omp parallel, nowait
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:19: error: expected an OpenMP clause, but found ','

index_variable(_I) :-
    %$omp do index(f(_I))
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:20: error: expected a variable, but found 'f(_I)'

map_modifier(_X) :-
    %$omp target map(sideways: _X)
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:22: error: expected a map modifier: always, close or present, but found 'sideways'

map_object(_X) :-
    %$omp target map(to: f(_X))
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:26: error: expected a variable or an array section, Variable(Lo .. Hi), but found 'f(_X)'

unclosed :-
    %$omp do schedule(static
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:29: error: expected an operator, ',' or ')'

empty :-
    %$omp
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:5: error: expected an OpenMP directive after '%$omp'

% As in mmc, an operator before an infix operator must be in parentheses.
reduction_list(_X) :-
    %$omp do reduction(+ : _X)
    true.
% CHECK: omp-errors.m:[[@LINE-2]]:24: error: the operator '+' must be in parentheses to be used as an operand

%$omp parallel
before_head.
% CHECK: omp-errors.m:[[@LINE-2]]:1: warning: an OpenMP directive applies only to a goal; it is ignored

in_dcg -->
    %$omp parallel
    [].
% CHECK: omp-errors.m:[[@LINE-2]]:5: warning: an OpenMP directive applies only to a goal; it is ignored

in_data_term(X) :-
    X =
    %$omp parallel
        1.
% CHECK: omp-errors.m:[[@LINE-2]]:5: warning: an OpenMP directive applies only to a goal; it is ignored

% The items with errors are left out.
% CHECK-NOT: OmpGoal
% CHECK:     | | | name: SymName before_head
% CHECK-NOT: OmpGoal
