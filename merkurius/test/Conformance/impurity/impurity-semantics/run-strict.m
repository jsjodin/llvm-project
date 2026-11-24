% MANUAL: Impurity semantics
% Impure goals run when they are reached, even when their outputs are
% unused, the conjunction then fails, or they are in a condition or a
% negation.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:before
% OUTPUT-NEXT:t no
% OUTPUT-NEXT:u
% OUTPUT-NEXT:u no
% OUTPUT-NEXT:condition
% OUTPUT-NEXT:else
% OUTPUT-NEXT:negation
% OUTPUT-NEXT:not
% OUTPUT-NEXT:w1
% OUTPUT-NEXT:w2
% OUTPUT-NEXT:end
% OUTPUT-NOT:{{.}}

:- module run_strict.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pragma foreign_decl("C", local, "#include <stdio.h>").
:- impure pred say(string::in, int::out) is det.
:- pragma foreign_proc("C", say(S::in, X::out), [will_not_call_mercury],
    "printf(""%s\\n"", S); fflush(stdout); X = 0;").
:- impure pred say_fail(string::in) is semidet.
:- pragma foreign_proc("C", say_fail(S::in), [will_not_call_mercury],
    "printf(""%s\\n"", S); fflush(stdout); SUCCESS_INDICATOR = 0;").
:- pred t(int::in) is semidet.
:- pragma promise_pure(t/1).
t(X) :- impure say("before", _), X > 5, impure say("after", _).
:- pred u is semidet.
:- pragma promise_pure(u/0).
u :- impure say("u", _), fail.
:- pred v is det.
:- pragma promise_pure(v/0).
v :-
    ( if impure say_fail("condition") then true else impure say("else", _) ),
    ( if not impure say_fail("negation") then impure say("not", _) else true ).
:- pred w is det.
:- pragma promise_pure(w/0).
w :- impure say("w1", _), impure say("w2", _).

main(!IO) :-
    ( if t(3) then io.write_string("t yes\n", !IO) else io.write_string("t no\n", !IO) ),
    ( if u then io.write_string("u yes\n", !IO) else io.write_string("u no\n", !IO) ),
    v, w,
    io.write_string("end\n", !IO).
