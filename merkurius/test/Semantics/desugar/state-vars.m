% State variables become numbered versions, S!0, S!1, ..., following the
% reference manual's examples ("State variables"). The module compiles with
% mmc.
%
% RUN: merkc -fc1 -fdebug-dump-semantic-form %s | FileCheck %s

:- module state_vars.
:- interface.
:- import_module io, list.

:- pred main(io::di, io::uo) is det.
:- pred fold2(pred(T, A, A, B, B), list(T), A, A, B, B).
:- mode fold2(pred(in, in, out, in, out) is det, in, in, out, in, out)
    is det.
:- pred iterate_while2(pred(A, B), pred(A, A, B, B), A, A, B, B).
:- mode iterate_while2(pred(in, in) is semidet,
    pred(in, out, in, out) is det, in, out, in, out) is det.
:- pred compute_out(int::in, int::out) is det.

:- implementation.
:- import_module int.

main(!IO) :-
    io.write_string("The 100th prime is ", !IO),
    X = 541,
    io.write_int(X, !IO),
    io.nl(!IO).
% CHECK:      Predicate main(IO!0, IO!3)
% CHECK-NEXT: | Conj #5
% CHECK-NEXT: | | Unify #0 $1 = "The 100th prime is "
% CHECK-NEXT: | | Call #1 io.write_string($1, IO!0, IO!1)
% CHECK-NEXT: | | Unify #2 X = 541
% CHECK-NEXT: | | Call #3 io.write_int(X, IO!1, IO!2)
% CHECK-NEXT: | | Call #4 io.nl(IO!2, IO!3)

fold2(_, [], !A, !B).
% CHECK-NEXT: Predicate fold2(_$2, $3, A!0, A!1, B!0, B!1)
% CHECK-NEXT: | Conj #3
% CHECK-NEXT: | | Unify #0 $3 = []
% CHECK-NEXT: | | Unify #1 A!1 = A!0
% CHECK-NEXT: | | Unify #2 B!1 = B!0
fold2(P, [X | Xs], !A, !B) :-
    P(X, !A, !B),
    fold2(P, Xs, !A, !B).
% CHECK-NEXT: Predicate fold2(P, $3, A!0, A!2, B!0, B!2)
% CHECK-NEXT: | Conj #3
% CHECK-NEXT: | | Unify #0 $3 = [X | Xs]
% CHECK-NEXT: | | HOCall #1 P(X, A!0, A!1, B!0, B!1)
% CHECK-NEXT: | | Call #2 fold2(P, Xs, A!1, A!2, B!1, B!2)

% The branches of an if-then-else end at one new version of each state
% variable that they leave at different versions.
iterate_while2(Test, Update, !A, !B) :-
    ( if Test(!.A, !.B) then
        Update(!A, !B),
        iterate_while2(Test, Update, !A, !B)
    else
        true
    ).
% CHECK-NEXT: Predicate iterate_while2(Test, Update, A!0, A!3, B!0, B!3)
% CHECK-NEXT: | IfThenElse #9
% CHECK-NEXT: | | HOCall #0 Test(A!0, B!0)
% CHECK-NEXT: | | Conj #8
% CHECK-NEXT: | | | HOCall #1 Update(A!0, A!1, B!0, B!1)
% CHECK-NEXT: | | | Call #2 iterate_while2(Test, Update, A!1, A!2, B!1, B!2)
% CHECK-NEXT: | | | Unify #3 A!3 = A!2
% CHECK-NEXT: | | | Unify #5 B!3 = B!2
% CHECK-NEXT: | | Conj #7
% CHECK-NEXT: | | | Unify #4 A!3 = A!0
% CHECK-NEXT: | | | Unify #6 B!3 = B!0

% some [!State] introduces a state variable with no value at first.
compute_out(In, Out) :-
    some [!State] (
        !:State = In,
        !:State = !.State + 1,
        Out = !.State
    ).
% CHECK-NEXT: Predicate compute_out(In, Out)
% CHECK-NEXT: | Scope #5 Some
% CHECK-NEXT: | | Conj #4
% CHECK-NEXT: | | | Unify #0 State!0 = In
% CHECK-NEXT: | | | Unify #2 State!1 = '+'(State!0, $4)
% CHECK-NEXT: | | | Unify #1 $4 = 1
% CHECK-NEXT: | | | Unify #3 Out = State!1
