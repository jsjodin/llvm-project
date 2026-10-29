% %$omp directives are tokens only with -fopenmp; without it they are
% comments, so the module is the sequential program that mmc compiles.
%
% RUN: merkc -fc1 -fdebug-unparse-tokens %s | cmp - %s
% RUN: merkc -fc1 -fopenmp -fdebug-unparse-tokens %s | cmp - %s
% RUN: merkc -fc1 -fopenmp -fdebug-dump-tokens %s | FileCheck %s
% RUN: merkc -fc1 -fdebug-dump-tokens %s | FileCheck %s --check-prefix=SEQ

:- module directives.
:- interface.
:- import_module array, int.

:- pred sum_squares(array(int)::in, int::in, int::out) is det.

:- implementation.

sum_squares(A, N, Sum) :-
    %$omp parallel do
    %$omp&    reduction(+)
    int.fold_up(add_square(A), 0, N, 0, Sum).
% CHECK:      [[@LINE-3]]:5 directive
% CHECK-NEXT:   [[@LINE-4]]:11 name parallel
% CHECK-NEXT:   [[@LINE-5]]:20 name do
% CHECK-NEXT:   [[@LINE-5]]:15 name reduction
% CHECK-NEXT:   [[@LINE-6]]:24 open_ct (
% CHECK-NEXT:   [[@LINE-7]]:25 name +
% CHECK-NEXT:   [[@LINE-8]]:26 close )
% CHECK-NEXT:   [[@LINE-9]]:27 eof
% CHECK-NEXT: [[@LINE-9]]:5 name int
% SEQ-NOT: {{^[0-9]+:[0-9]+}} directive{{$}}
% SEQ-NOT: name parallel

% A directive inside a string is part of the string.
:- func text = string.
text = "
    %$omp parallel do
".
% CHECK: string "\n    %$omp parallel do\n"

:- pred add_square(array(int)::in, int::in, int::in, int::out) is det.

add_square(A, I, !Acc) :-
    array.unsafe_lookup(A, I, X),
    !:Acc = !.Acc + X * X.
