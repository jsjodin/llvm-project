% MANUAL: Impurity semantics
% A unification of a head variable waits past an impure goal for the
% values it binds, as mmc lets it, so that an impure procedure has outputs:
% in the head, in the body, and of a term of a clause's head. A term nested
% in a head term is not, as mmc rejects it.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{[1], {2, 3}, [13], [33]}
% OUTPUT-NOT:{{.}}

:- module run_head_unifications_wait.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pragma foreign_decl("C", local, "static long state = 0;").
:- impure pred set(int::in) is det.
:- pragma foreign_proc("C", set(X::in), [will_not_call_mercury],
    "state = X;").
:- semipure pred get(int::out) is det.
:- pragma foreign_proc("C", get(X::out),
    [promise_semipure, will_not_call_mercury], "X = state;").
:- impure func bump(int) = int.
:- pragma foreign_proc("C", bump(X::in) = (Y::out), [will_not_call_mercury],
    "state += X; Y = state;").
:- semipure func peek = int.
:- pragma foreign_proc("C", peek = (X::out),
    [promise_semipure, will_not_call_mercury], "X = state;").

:- impure pred head(list(int)::out) is det.
head([X]) :- impure X = bump(1).
:- impure pred body({int, int}::out) is det.
body(R) :- R = {X, Y}, impure X = bump(1), impure Y = bump(1).
:- impure pred clauses(int::in, list(int)::out) is semidet.
clauses(0, [X]) :- impure X = bump(10).
clauses(1, [X]) :- impure X = bump(20).

main(!IO) :-
    promise_pure (
        impure head(A),
        impure body(B),
        ( if impure clauses(0, C0) then C = C0 else C = [] ),
        ( if impure clauses(1, D0) then D = D0 else D = [] )
    ),
    io.write({A, B, C, D}, !IO), io.nl(!IO).
