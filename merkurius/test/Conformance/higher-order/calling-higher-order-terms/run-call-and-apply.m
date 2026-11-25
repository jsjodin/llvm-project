% MANUAL: Calling higher-order terms
% call/N appends its arguments to the closure's and calls it, apply/N
% applies a function term, and solutions/2 takes a closure: the manual's
% examples, with call/1 of a closure with no arguments left.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:6
% OUTPUT-NEXT:9
% OUTPUT-NEXT:[2, 4, 6]
% OUTPUT-NEXT:positive
% OUTPUT-NEXT:{{\[\[}}1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
% OUTPUT-NOT:{{.}}

:- module run_call_and_apply.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- pred sum(list(int)::in, int::out) is det.
sum(List, Sum) :-
    list.foldl((pred(E::in, A0::in, A::out) is det :- A = A0 + E), List, 0,
        Sum).

:- func scalar_product(int, list(int)) = list(int).
scalar_product(Num, List) = list.map((func(E) = Num * E), List).

:- pred positive(int::in) is semidet.
positive(X) :-
    X > 0.

main(!IO) :-
    Sum123 = sum([1, 2, 3]),
    call(Sum123, Result),
    io.write_line(Result, !IO),
    Sum = sum,
    call(Sum, [4, 5], R2),
    io.write_line(R2, !IO),
    Double = scalar_product(2),
    List = apply(Double, [1, 2, 3]),
    io.write_line(List, !IO),
    ( if call(positive(3)) then
        io.write_string("positive\n", !IO)
    else
        io.write_string("not positive\n", !IO)
    ),
    solutions(list.perm([3, 1, 2]), L),
    io.write_line(L, !IO).
