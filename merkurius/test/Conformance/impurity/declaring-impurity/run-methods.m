% MANUAL: Declaring impurity
% Type class methods declared impure and semipure, with an instance whose
% semipure method is implemented by a pure function, which is at least as
% pure.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[77, 2]
% OUTPUT-NOT:{{.}}

:- module run_methods.
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

:- typeclass counter(T) where [
    impure pred incr(T::in) is det,
    semipure func value(T) = int
].
:- type c ---> c.
:- instance counter(c) where [
    pred(incr/1) is c_incr,
    func(value/1) is c_value
].
:- impure pred c_incr(c::in) is det.
c_incr(_) :- semipure get(X), impure set(X + 1).
:- func c_value(c) = int.
c_value(_) = 77.

main(!IO) :-
    promise_pure (
        impure incr(c), impure incr(c),
        semipure V = value(c), semipure get(S)
    ),
    io.write([V, S], !IO), io.nl(!IO).
