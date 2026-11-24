% MANUAL: Purity annotations on higher-order types
% Closures of impure predicates and functions, made from their names and
% by partial application, in lists of impure higher-order types.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:110
% OUTPUT-NEXT:[112, 114]
% OUTPUT-NOT:{{.}}

:- module run_closures.
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

:- pred run_all(list(impure pred(int))::in(list(pred(in) is det)), int::in)
    is det.
:- pragma promise_pure(run_all/2).
run_all([], _).
run_all([P | Ps], X) :- impure P(X), run_all(Ps, X).
:- func apply_all(list(impure func(int) = int), int) = list(int).
:- mode apply_all(in(list(func(in) = out is det)), in) = out is det.
:- pragma promise_pure(apply_all/2).
apply_all([], _) = [].
apply_all([F | Fs], X) = [Y | Ys] :-
    impure Y = impure_apply(F, X),
    Ys = apply_all(Fs, X).
:- impure pred add(int::in, int::in) is det.
add(A, B) :- semipure get(S), impure set(S + A + B).

main(!IO) :-
    Set = set, Add = add(100),
    run_all([Set, Add], 5),
    promise_pure ( semipure get(S) ),
    io.write_int(S, !IO), io.nl(!IO),
    Bump = bump,
    io.write(apply_all([Bump, Bump], 2), !IO), io.nl(!IO).
