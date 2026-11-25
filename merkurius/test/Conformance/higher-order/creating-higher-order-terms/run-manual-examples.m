% MANUAL: Creating higher-order terms
% The manual's ways to create higher-order terms: a lambda expression or
% a single-mode predicate's name, the three equivalent function terms with
% the default modes and determinism, currying a predicate and a function,
% currying list.foldl with io.write, a DCG lambda expression, a
% zero-arity function term, and a lambda expression or forwarding
% predicate in place of a builtin.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{6, 9}
% OUTPUT-NEXT:{{\[\[}}2], [3], [4]]
% OUTPUT-NEXT:6
% OUTPUT-NEXT:[2, 4, 6]
% OUTPUT-NEXT:123
% OUTPUT-NEXT:The strings are: ab
% OUTPUT-NEXT:2
% OUTPUT-NEXT:42
% OUTPUT-NEXT:{[1, 3], [1, 3]}
% OUTPUT-NOT:{{.}}

:- module run_manual_examples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred sum(list(int), int).
:- mode sum(in, out) is det.
sum(List, Sum) :-
    list.foldl((pred(E::in, A0::in, A::out) is det :- A = A0 + E), List, 0,
        Sum).

:- func scalar_product(int, list(int)) = list(int).
:- mode scalar_product(in, in) = out is det.
scalar_product(Num, List) = list.map((func(E) = Num * E), List).

:- func foo = int.
foo = 42.

:- pred not_equal(T::in, T::in) is semidet.
not_equal(X, Y) :-
    X \= Y.

main(!IO) :-
    X = (pred(List::in, Length::out) is det :- sum(List, Length)),
    Y = sum,
    X([1, 2, 3], SX),
    Y([4, 5], SY),
    io.write_line({SX, SY}, !IO),
    FX = (func(Num, List) = NewList :- NewList = scalar_product(Num, List)),
    FY = (func(Num::in, List::in) = (NewList::out) is det
            :- NewList = scalar_product(Num, List)),
    FZ = scalar_product,
    io.write_line([FX(2, [1]), FY(3, [1]), FZ(4, [1])], !IO),
    Sum123 = sum([1, 2, 3]),
    Sum123(S123),
    io.write_line(S123, !IO),
    Double = scalar_product(2),
    io.write_line(Double([1, 2, 3]), !IO),
    P = list.foldl(io.write),
    P([1, 2, 3], !IO),
    io.nl(!IO),
    Pred = (pred(Strings::in, Num::out, di, uo) is det -->
        io.write_string("The strings are: "),
        { list.length(Strings, Num) },
        io.write_strings(Strings),
        io.nl
    ),
    Pred(["a", "b"], N, !IO),
    io.write_line(N, !IO),
    F0 = ((func) = foo),
    io.write_line(apply(F0), !IO),
    list.filter((pred(E::in) is semidet :- E \= 2), [1, 2, 3], L1),
    list.filter(not_equal(2), [1, 2, 3], L2),
    io.write_line({L1, L2}, !IO).
