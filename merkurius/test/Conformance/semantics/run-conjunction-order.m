% MANUAL: Semantics
% The strict sequential semantics reorders a conjunction minimally: it runs
% the first conjunct that is mode-correct, then the first of the rest that
% is now mode-correct, and so on. Each goal throws the name of the conjunct
% that runs first; one that a failing test precedes fails instead.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:"a"
% OUTPUT-NEXT:"a"
% OUTPUT-NEXT:"c"
% OUTPUT-NEXT:"b"
% OUTPUT-NEXT:"b"
% OUTPUT-NEXT:"binding"
% OUTPUT-NEXT:failed
% OUTPUT-NEXT:"before a failure"
% OUTPUT-NEXT:"negated"
% OUTPUT-NOT:{{.}}

:- module run_conjunction_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string, univ.

% Throws S, but is det: no pass may treat a call of it as erroneous.
:- func th(string, int) = int.
th(S, N) = ( if N > -1000 then throw(S) else N ).

:- pred show(exception_result(int)::in, io::di, io::uo) is det.
show(succeeded(N), !IO) :- io.format("succeeded %d\n", [i(N)], !IO).
show(failed, !IO) :- io.write_string("failed\n", !IO).
show(exception(E), !IO) :- io.write(univ_value(E), !IO), io.nl(!IO).

% B = 1 runs first, then A, the first conjunct it makes mode-correct.
:- pred made_correct(int::out) is det.
made_correct(R) :- A = th("a", B), B = 1, C = th("c", 1), R = A + C.

% C waits for B; A is mode-correct from the start.
:- pred first_correct(int::out) is det.
first_correct(R) :- C = th("c", B), A = th("a", 1), B = 1, R = A + C.

% R = A + C waits for both; C comes before A.
:- pred sum_first(int::out) is det.
sum_first(R) :- R = A + C, C = th("c", 1), A = th("a", 1).

% Both A and C wait for B, which throws first.
:- pred shared_input(int::out) is det.
shared_input(R) :-
    A = th("a", B), C = th("c", B), B = th("b", 1), R = A + C.

% An if-then-else waits, as one goal, for what its condition needs.
:- pred whole_goal(int::out) is det.
whole_goal(R) :-
    ( if A = th("a", B) then R = A else R = 0 ),
    B = th("b", 1).

% A test waits for the binding it needs.
:- pred test_waits(int::out) is semidet.
test_waits(R) :- X > 5, X = th("binding", 1), R = X.

% A failing test before a throw fails.
:- pred fail_first(int::out) is semidet.
fail_first(R) :- 1 = 2, R = th("after a failure", 1).

% A throw before a failing test throws.
:- pred throw_first(int::out) is semidet.
throw_first(R) :- R = th("before a failure", 1), 1 = 2.

% Left to right within a negation.
:- pred negated(int::out) is semidet.
negated(R) :- not ( X = th("negated", 1), X > 0 ), R = 1.

main(!IO) :-
    try(made_correct, R1), show(R1, !IO),
    try(first_correct, R2), show(R2, !IO),
    try(sum_first, R3), show(R3, !IO),
    try(shared_input, R4), show(R4, !IO),
    try(whole_goal, R5), show(R5, !IO),
    try(test_waits, R6), show(R6, !IO),
    try(fail_first, R7), show(R7, !IO),
    try(throw_first, R8), show(R8, !IO),
    try(negated, R9), show(R9, !IO).
