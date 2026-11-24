% MANUAL: Purity annotations on higher-order types
% A pure closure converted to an impure higher-order type by an impure
% lambda expression that calls it.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:8
% OUTPUT-NOT:{{.}}

:- module run_converted.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- func twice(impure pred(int)) = int.
:- mode twice(in(pred(out) is det)) = out is det.
:- pragma promise_pure(twice/1).
twice(P) = X :- impure P(X1), impure P(X2), X = X1 + X2.

main(!IO) :-
    Pure = (pred(Y::out) is det :- Y = 4),
    Impure = (impure pred(Y::out) is det :- Pure(Y)),
    io.write_int(twice(Impure), !IO), io.nl(!IO).
