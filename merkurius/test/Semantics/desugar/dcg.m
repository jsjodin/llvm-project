% DCG rules become clauses with two more arguments, the first and last
% versions of a hidden state variable, printed as dcg!N (reference manual,
% "DCG-goals"). The module compiles with mmc.
%
% RUN: merkc -fc1 -fdebug-dump-semantic-form %s | FileCheck %s

:- module dcg.
:- interface.
:- import_module char, list.

:- pred digits(list(char)::out, list(char)::in, list(char)::out) is det.
:- pred sign(int::out, list(char)::in, list(char)::out) is det.

:- implementation.

digits(Ds) -->
    ( if [D], { char.is_digit(D) } then
        digits(Ds0),
        { Ds = [D | Ds0] }
    else
        { Ds = [] }
    ).
% CHECK:      Predicate digits(Ds, dcg!0, dcg!3)
% CHECK-NEXT: | IfThenElse #10
% CHECK-NEXT: | | Conj #2
% CHECK-NEXT: | | | Unify #0 dcg!0 = [D | dcg!1]
% CHECK-NEXT: | | | Call #1 char.is_digit(D)
% CHECK-NEXT: | | Conj #9
% CHECK-NEXT: | | | Call #3 digits(Ds0, dcg!1, dcg!2)
% CHECK-NEXT: | | | Unify #4 Ds = [D | Ds0]
% CHECK-NEXT: | | | Unify #6 dcg!3 = dcg!2
% CHECK-NEXT: | | Conj #8
% CHECK-NEXT: | | | Unify #5 Ds = []
% CHECK-NEXT: | | | Unify #7 dcg!3 = dcg!0

% Nested if-then-elses: the input matches of the conditions are seen in
% their then parts only, and every branch ends at the last version.
sign(S) -->
    ( ['-'] ->
        { S = -1 }
    ; ['+'] ->
        { S = 1 }
    ;
        [],
        { S = 1 }
    ).
% CHECK:      Predicate sign(S, dcg!0, dcg!4)
% CHECK-NEXT: | IfThenElse #18
% CHECK-NEXT: | | Conj #2
% CHECK-NEXT: | | | Unify #1 dcg!0 = [$3 | dcg!1]
% CHECK-NEXT: | | | Unify #0 $3 = '-'
% CHECK-NEXT: | | Conj #17
% CHECK-NEXT: | | | Unify #3 S = -1
% CHECK-NEXT: | | | Unify #14 dcg!4 = dcg!1
% CHECK-NEXT: | | Conj #16
% CHECK-NEXT: | | | IfThenElse #13
% CHECK-NEXT: | | | | Conj #6
% CHECK-NEXT: | | | | | Unify #5 dcg!0 = [$5 | dcg!2]
% CHECK-NEXT: | | | | | Unify #4 $5 = '+'
% CHECK-NEXT: | | | | Conj #12
% CHECK-NEXT: | | | | | Unify #7 S = 1
% CHECK-NEXT: | | | | | Unify #9 dcg!3 = dcg!2
% CHECK-NEXT: | | | | Conj #11
% CHECK-NEXT: | | | | | Unify #8 S = 1
% CHECK-NEXT: | | | | | Unify #10 dcg!3 = dcg!0
% CHECK-NEXT: | | | Unify #15 dcg!4 = dcg!3
