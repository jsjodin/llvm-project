% try goals (reference manual, "Exception handling"), atomic goals, which
% the manual does not describe and which follow mmc 22.01.9, and state
% variables in a scope's list of variables. The module compiles with mmc.
%
% RUN: merkc -fc1 -fdebug-dump-semantic-form %s | FileCheck %s

:- module try_atomic.
:- interface.
:- import_module io, list, stm_builtin.

:- pred p_carefully(io::di, io::uo) is cc_multi.
:- pred t(int::in, int::out) is cc_multi.
:- pred take(stm_var(int)::in, int::out, io::di, io::uo) is cc_multi.
:- pred sum_some(list(int)::in, int::in, int::out) is det.

:- implementation.
:- import_module exception, int.

:- pred p(int::out, io::di, io::uo) is det.
p(1, !IO).

% The manual's example ("Exception handling"). The io(!IO) state continues
% from the version after the goal in every branch.
p_carefully(!IO) :-
    (try [io(!IO)] (
        io.write_string("Calling p\n", !IO),
        p(Output, !IO)
    )
    then
        io.write_string("p returned: ", !IO),
        io.write(Output, !IO),
        io.nl(!IO)
    catch S ->
        io.write_string("p threw a string: ", !IO),
        io.write_string(S, !IO),
        io.nl(!IO)
    catch 42 ->
        io.write_string("p threw 42\n", !IO)
    catch_any Other ->
        io.write_string("p threw something: ", !IO),
        io.write(Other, !IO),
        % Rethrow the value.
        throw(Other)
    ).
% CHECK:      Predicate p_carefully(IO!0, IO!12)
% CHECK-NEXT: | Try #29 io [IO!0, IO!2]
% CHECK-NEXT: | | Conj #3
% CHECK-NEXT: | | | Unify #0 $1 = "Calling p\n"
% CHECK-NEXT: | | | Call #1 io.write_string($1, IO!0, IO!1)
% CHECK-NEXT: | | | Call #2 p(Output, IO!1, IO!2)
% CHECK-NEXT: | | then
% CHECK-NEXT: | | | Conj #25
% CHECK-NEXT: | | | | Unify #4 $5 = "p returned: "
% CHECK-NEXT: | | | | Call #5 io.write_string($5, IO!2, IO!3)
% CHECK-NEXT: | | | | Call #6 io.write(Output, IO!3, IO!4)
% CHECK-NEXT: | | | | Call #7 io.nl(IO!4, IO!5)
% CHECK-NEXT: | | | | Unify #21 IO!12 = IO!5
% CHECK-NEXT: | | catch S
% CHECK-NEXT: | | | Conj #26
% CHECK-NEXT: | | | | Unify #9 $10 = "p threw a string: "
% CHECK-NEXT: | | | | Call #10 io.write_string($10, IO!2, IO!6)
% CHECK-NEXT: | | | | Call #11 io.write_string(S, IO!6, IO!7)
% CHECK-NEXT: | | | | Call #12 io.nl(IO!7, IO!8)
% CHECK-NEXT: | | | | Unify #22 IO!12 = IO!8
% CHECK-NEXT: | | catch $14
% CHECK-NEXT: | | | match
% CHECK-NEXT: | | | | Unify #13 $14 = 42
% CHECK-NEXT: | | | Conj #27
% CHECK-NEXT: | | | | Unify #14 $15 = "p threw 42\n"
% CHECK-NEXT: | | | | Call #15 io.write_string($15, IO!2, IO!9)
% CHECK-NEXT: | | | | Unify #23 IO!12 = IO!9
% CHECK-NEXT: | | catch_any Other
% CHECK-NEXT: | | | Conj #28
% CHECK-NEXT: | | | | Unify #17 $18 = "p threw something: "
% CHECK-NEXT: | | | | Call #18 io.write_string($18, IO!2, IO!10)
% CHECK-NEXT: | | | | Call #19 io.write(Other, IO!10, IO!11)
% CHECK-NEXT: | | | | Call #20 throw(Other)
% CHECK-NEXT: | | | | Unify #24 IO!12 = IO!11

% As in mmc, the goal's updates of other state variables are seen in the
% then part only: a catch sees the version before the goal.
t(!S) :-
    ( try []
        ( !:S = !.S + 1,
          ( if !.S > 5 then throw(!.S) else true )
        )
    then
        !:S = !.S * 100
    catch E ->
        !:S = !.S * 1000 + E
    ).
% CHECK:      Predicate t(S!0, S!4)
% CHECK-NEXT: | Try #19
% CHECK-NEXT: | | Conj #8
% CHECK-NEXT: | | | Unify #1 S!1 = '+'(S!0, $2)
% CHECK-NEXT: | | | Unify #0 $2 = 1
% CHECK-NEXT: | | | IfThenElse #7
% CHECK-NEXT: | | | | Conj #4
% CHECK-NEXT: | | | | | Unify #2 $3 = 5
% CHECK-NEXT: | | | | | Call #3 '>'(S!1, $3)
% CHECK-NEXT: | | | | Call #5 throw(S!1)
% CHECK-NEXT: | | | | Conj #6
% CHECK-NEXT: | | then
% CHECK-NEXT: | | | Conj #17
% CHECK-NEXT: | | | | Unify #10 S!2 = '*'(S!1, $5)
% CHECK-NEXT: | | | | Unify #9 $5 = 100
% CHECK-NEXT: | | | | Unify #15 S!4 = S!2
% CHECK-NEXT: | | catch E
% CHECK-NEXT: | | | Conj #18
% CHECK-NEXT: | | | | Unify #14 S!3 = '+'($8, E)
% CHECK-NEXT: | | | | Unify #13 $8 = '*'(S!0, $9)
% CHECK-NEXT: | | | | Unify #12 $9 = 1000
% CHECK-NEXT: | | | | Unify #16 S!4 = S!3

% An STM transaction: the outer state is hidden inside, and each
% alternative has inner state of its own.
take(V, Y, !IO) :-
    atomic [outer(!IO), inner(!STM)] (
        read_stm_var(V, X, !STM),
        ( if X > 0 then
            write_stm_var(V, X - 1, !STM), Y = X
        else
            retry(!.STM)
        )
    or_else
        read_stm_var(V, _, !STM), Y = 0
    ).
% CHECK:      Predicate take(V, Y, IO!0, IO!1)
% CHECK-NEXT: | Atomic #18 outer [IO!0, IO!1]
% CHECK-NEXT: | | inner [STM!0, STM!3]
% CHECK-NEXT: | | | Conj #16
% CHECK-NEXT: | | | | Call #0 read_stm_var(V, X, STM!0, STM!1)
% CHECK-NEXT: | | | | IfThenElse #13
% CHECK-NEXT: | | | | | Conj #3
% CHECK-NEXT: | | | | | | Unify #1 $7 = 0
% CHECK-NEXT: | | | | | | Call #2 '>'(X, $7)
% CHECK-NEXT: | | | | | Conj #12
% CHECK-NEXT: | | | | | | Unify #5 $8 = '-'(X, $9)
% CHECK-NEXT: | | | | | | Unify #4 $9 = 1
% CHECK-NEXT: | | | | | | Call #6 write_stm_var(V, $8, STM!1, STM!2)
% CHECK-NEXT: | | | | | | Unify #7 Y = X
% CHECK-NEXT: | | | | | | Unify #9 STM!3 = STM!2
% CHECK-NEXT: | | | | | Conj #11
% CHECK-NEXT: | | | | | | Call #8 retry(STM!1)
% CHECK-NEXT: | | | | | | Unify #10 STM!3 = STM!1
% CHECK-NEXT: | | or_else inner [STM!4, STM!5]
% CHECK-NEXT: | | | Conj #17
% CHECK-NEXT: | | | | Call #14 read_stm_var(V, _$13, STM!4, STM!5)
% CHECK-NEXT: | | | | Unify #15 Y = 0

% !:S in a promise's list is the version after the goal.
sum_some(L, !S) :-
    promise_equivalent_solutions [!:S] (
        list.member(X, L), !:S = !.S + X
    ;
        true
    ).
% CHECK:      Predicate sum_some(L, S!0, S!2)
% CHECK-NEXT: | Scope #6 PromiseSolutions [S!2]
% CHECK-NEXT: | | Disj #5
% CHECK-NEXT: | | | Conj #4
% CHECK-NEXT: | | | | Call #0 list.member(X, L)
% CHECK-NEXT: | | | | Unify #1 S!1 = '+'(S!0, X)
% CHECK-NEXT: | | | | Unify #2 S!2 = S!1
% CHECK-NEXT: | | | Unify #3 S!2 = S!0
