% Implicit quantification (reference manual, "Implicit quantification"):
% variables renamed apart, and each goal's nonlocal variables, after "~".
% The module compiles with mmc, which gives the same warnings.
%
% RUN: merkc -fc1 -fdebug-dump-nonlocals %s 2>/dev/null | FileCheck %s
% RUN: merkc -fc1 -fdebug-dump-nonlocals %s 2>&1 >/dev/null | FileCheck --check-prefix=WARN %s

:- module quantification.
:- interface.
:- import_module io, list, stm_builtin.

:- pred disjuncts(int::in) is semidet.
:- pred overlapping(int::in, int::out) is semidet.
:- pred shadowed(list(int)::in, list(int)::out) is semidet.
:- pred condition(list(int)::in, int::out) is multi.
:- pred lambdas(int::out) is det.
:- pred io_try(io::di, io::uo) is cc_multi.
:- pred transaction(stm_var(int)::in, io::di, io::uo) is det.

:- implementation.
:- import_module exception, int, string.

% X is local to each disjunct, so it is two variables, of two types.
disjuncts(N) :-
    ( N = 1, X = 2, X > 0
    ; N = 2, X = "s", X \= ""
    ).
% CHECK:      Predicate disjuncts(N)
% CHECK-NEXT: | Disj #10 ~ [N]
% CHECK-NEXT: | | Conj #8 ~ [N]
% CHECK-NEXT: | | | Unify #0 N = 1 ~ [N]
% CHECK-NEXT: | | | Unify #1 X = 2 ~ [X]
% CHECK-NEXT: | | | Unify #2 $2 = 0 ~ [$2]
% CHECK-NEXT: | | | Call #3 '>'(X, $2) ~ [X, $2]
% CHECK-NEXT: | | Conj #9 ~ [N]
% CHECK-NEXT: | | | Unify #4 N = 2 ~ [N]
% CHECK-NEXT: | | | Unify #5 X$3 = "s" ~ [X$3]
% CHECK-NEXT: | | | Not #7 ~ [X$3]
% CHECK-NEXT: | | | | Unify #6 X$3 = "" ~ [X$3]

% some quantifies a new X, apart from the one outside, which mmc warns of.
overlapping(N, M) :-
    X = N + 1,
    some [X] ( X = 3, X > 2 ),
    M = X.
% WARN: quantification.m:[[@LINE-2]]:{{[0-9]+}}: warning: variable X has overlapping scopes
% CHECK:      Predicate overlapping(N, M)
% CHECK-NEXT: | Conj #8 ~ [N, M]
% CHECK-NEXT: | | Unify #1 X = '+'(N, $3) ~ [N, X, $3]
% CHECK-NEXT: | | Unify #0 $3 = 1 ~ [$3]
% CHECK-NEXT: | | Scope #6 Some [X$5] ~ []
% CHECK-NEXT: | | | Conj #5 ~ []
% CHECK-NEXT: | | | | Unify #2 X$5 = 3 ~ [X$5]
% CHECK-NEXT: | | | | Unify #3 $4 = 2 ~ [$4]
% CHECK-NEXT: | | | | Call #4 '>'(X$5, $4) ~ [$4, X$5]
% CHECK-NEXT: | | Unify #7 M = X ~ [M, X]

% So is a lambda argument with the name of a variable outside it.
shadowed(L, M) :-
    X = 1,
    M = list.map((func(X) = X + 1), L),
    X > 0.
% WARN: quantification.m:[[@LINE-2]]:{{[0-9]+}}: warning: variable X has overlapping scopes
% CHECK:      Predicate shadowed(L, M)
% CHECK-NEXT: | Conj #8 ~ [L, M]
% CHECK-NEXT: | | Unify #0 X = 1 ~ [X]
% CHECK-NEXT: | | Unify #5 M = list.map($3, L) ~ [L, M, $3]
% CHECK-NEXT: | | Unify #4 $3 = func(X$4) = ($5) ~ [$3]
% CHECK-NEXT: | | | Conj #3 ~ [X$4, $5]
% CHECK-NEXT: | | | | Unify #2 $5 = '+'(X$4, $6) ~ [X$4, $5, $6]
% CHECK-NEXT: | | | | Unify #1 $6 = 1 ~ [$6]
% CHECK-NEXT: | | Unify #6 $7 = 0 ~ [$7]
% CHECK-NEXT: | | Call #7 '>'(X, $7) ~ [X, $7]

% At the top of a condition, some quantifies over the then part too; the
% else part's X is its own.
condition(L, R) :-
    ( if some [X] list.member(X, L) then R = X else X = 0, R = X ).
% CHECK:      Predicate condition(L, R)
% CHECK-NEXT: | IfThenElse #6 ~ [L, R]
% CHECK-NEXT: | | Scope #1 Some [X] ~ [L, X]
% CHECK-NEXT: | | | Call #0 list.member(X, L) ~ [L, X]
% CHECK-NEXT: | | Unify #2 R = X ~ [R, X]
% CHECK-NEXT: | | Conj #5 ~ [R]
% CHECK-NEXT: | | | Unify #3 X$3 = 0 ~ [X$3]
% CHECK-NEXT: | | | Unify #4 R = X$3 ~ [R, X$3]

% The goals of two lambda expressions are parallel.
lambdas(R) :-
    P = (pred(A::out) is det :- X = 1, A = X),
    Q = (pred(A::out) is det :- X = "s", A = string.length(X)),
    P(R0),
    Q(R1),
    R = R0 + R1.
% CHECK:      Predicate lambdas(R)
% CHECK-NEXT: | Conj #11 ~ [R]
% CHECK-NEXT: | | Unify #3 P = pred(A::out) is det ~ [P]
% CHECK-NEXT: | | | Conj #2 ~ [A]
% CHECK-NEXT: | | | | Unify #0 X = 1 ~ [X]
% CHECK-NEXT: | | | | Unify #1 A = X ~ [A, X]
% CHECK-NEXT: | | Unify #7 Q = pred(A$5::out) is det ~ [Q]
% CHECK-NEXT: | | | Conj #6 ~ [A$5]
% CHECK-NEXT: | | | | Unify #4 X$8 = "s" ~ [X$8]
% CHECK-NEXT: | | | | Unify #5 A$5 = string.length(X$8) ~ [A$5, X$8]
% CHECK-NEXT: | | HOCall #8 P(R0) ~ [P, R0]
% CHECK-NEXT: | | HOCall #9 Q(R1) ~ [Q, R1]
% CHECK-NEXT: | | Unify #10 R = '+'(R0, R1) ~ [R, R0, R1]

% The I/O state after a try goal is shared with every branch.
io_try(!IO) :-
    ( try [io(!IO)] io.write_string("a", !IO)
    then io.write_string("b", !IO)
    catch_any E -> io.write(E, !IO)
    ).
% CHECK:      Predicate io_try(IO!0, IO!4)
% CHECK-NEXT: | Try #11 io [IO!0, IO!1] ~ [IO!0, IO!4]
% CHECK-NEXT: | | Conj #2 ~ [IO!0, IO!1]
% CHECK-NEXT: | | | Unify #0 $1 = "a" ~ [$1]
% CHECK-NEXT: | | | Call #1 io.write_string($1, IO!0, IO!1) ~ [IO!0, $1, IO!1]
% CHECK-NEXT: | | then
% CHECK-NEXT: | | | Conj #9 ~ [IO!1, IO!4]
% CHECK-NEXT: | | | | Unify #3 $3 = "b" ~ [$3]
% CHECK-NEXT: | | | | Call #4 io.write_string($3, IO!1, IO!2) ~ [IO!1, $3, IO!2]
% CHECK-NEXT: | | | | Unify #7 IO!4 = IO!2 ~ [IO!2, IO!4]
% CHECK-NEXT: | | catch_any E
% CHECK-NEXT: | | | Conj #10 ~ [IO!1, E, IO!4]
% CHECK-NEXT: | | | | Call #6 io.write(E, IO!1, IO!3) ~ [IO!1, E, IO!3]
% CHECK-NEXT: | | | | Unify #8 IO!4 = IO!3 ~ [IO!3, IO!4]

% Each atomic alternative's inner state is its own.
transaction(V, !IO) :-
    atomic [outer(!IO), inner(!STM)] (
        read_stm_var(V, X, !STM), write_stm_var(V, X + 1, !STM)
    ).
% CHECK:      Predicate transaction(V, IO!0, IO!1)
% CHECK-NEXT: | Atomic #5 outer [IO!0, IO!1] ~ [IO!0, V, IO!1]
% CHECK-NEXT: | | inner [STM!0, STM!2]
% CHECK-NEXT: | | | Conj #4 ~ [V, STM!0, STM!2]
% CHECK-NEXT: | | | | Call #0 read_stm_var(V, X, STM!0, STM!1) ~ [V, STM!0, X, STM!1]
% CHECK-NEXT: | | | | Unify #2 $6 = '+'(X, $7) ~ [X, $6, $7]
% CHECK-NEXT: | | | | Unify #1 $7 = 1 ~ [$7]
% CHECK-NEXT: | | | | Call #3 write_stm_var(V, $6, STM!1, STM!2) ~ [V, STM!1, $6, STM!2]
