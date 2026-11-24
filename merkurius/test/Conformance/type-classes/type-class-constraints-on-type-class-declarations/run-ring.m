% MANUAL: Type class constraints on type class declarations
% The manual's ring class and its subclass euclidean: a predicate
% constrained by euclidean calls ring's methods too. int is used, not
% imported, as its plus would make the methods' calls ambiguous.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:12
% OUTPUT-NEXT:3
% OUTPUT-NEXT:0
% OUTPUT-NOT:{{.}}

:- module run_ring.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- use_module int.

:- typeclass ring(T) where [
    func zero = (T::out) is det,               % '+' identity
    func one = (T::out) is det,                % '*' identity
    func plus(T::in, T::in) = (T::out) is det, % '+'/2 (forward mode)
    func mult(T::in, T::in) = (T::out) is det, % '*'/2 (forward mode)
    func negative(T::in) = (T::out) is det     % '-'/1 (forward mode)
].

:- typeclass euclidean(T) <= ring(T) where [
    func div(T::in, T::in) = (T::out) is det,
    func mod(T::in, T::in) = (T::out) is det
].

:- instance ring(int) where [
    zero = 0,
    one = 1,
    plus(X, Y) = int.'+'(X, Y),
    mult(X, Y) = int.'*'(X, Y),
    negative(X) = int.'-'(X)
].

:- instance euclidean(int) where [
    div(X, Y) = int.'//'(X, Y),
    mod(X, Y) = int.rem(X, Y)
].

:- func gcd(T, T) = T <= euclidean(T).
gcd(A, B) = ( if B = zero then A else gcd(B, mod(A, B)) ).

:- func sum_mod(T, T, T) = T <= euclidean(T).
sum_mod(A, B, M) = mod(plus(A, B), M).

main(!IO) :-
    io.write_line(gcd(84, 36), !IO),
    io.write_line(sum_mod(7, 8, 4), !IO),
    io.write_line(plus(one, negative(one)) : int, !IO).
