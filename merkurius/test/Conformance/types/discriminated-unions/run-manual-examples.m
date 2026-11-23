% MANUAL: Discriminated unions
% The manual's examples: an enumeration, constructors with arguments, a
% record with labelled fields, a recursive type, a polymorphic list and an
% operator constructor; values of each are built, switched on and written.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[apple, orange, banana, pear]
% OUTPUT-NEXT:[foo(1), bar("b")]
% OUTPUT-NEXT:employee("Ann", 41, "R&D")
% OUTPUT-NEXT:41
% OUTPUT-NEXT:3
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:3
% OUTPUT-NEXT:{1, "one"}
% OUTPUT-NEXT:107
% OUTPUT-NOT:{{.}}

:- module run_manual_examples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type fruit
    --->    apple
    ;       orange
    ;       banana
    ;       pear.

:- type strange
    --->    foo(int)
    ;       bar(string).

:- type employee
    --->    employee(
                name        :: string,
                age         :: int,
                department  :: string
            ).

:- type tree
    --->    empty
    ;       leaf(int)
    ;       branch(tree, tree).

:- type my_list(T)
    --->    []
    ;       [T | my_list(T)].

:- type my_pair(T1, T2)
    --->    T1 - T2.

main(!IO) :-
    io.write_line([apple, orange, banana, pear] : list(fruit), !IO),
    io.write_line([foo(1), bar("b")] : list(strange), !IO),
    E = employee("Ann", 41, "R&D"),
    io.write_line(E, !IO),
    io.write_line(E ^ age, !IO),
    T = branch(leaf(1), branch(empty, leaf(2))),
    io.write_line(sum(T), !IO),
    L = [1, 2, 3] : my_list(int),
    io.write_line(L, !IO),
    io.write_line(len(L), !IO),
    P = (1 - "one") : my_pair(int, string),
    P = N - S,
    io.write_line({N, S}, !IO),
    io.write_line(list_strange([foo(3), bar("x"), foo(4)] : list(strange)),
        !IO).

:- func sum(tree) = int.

sum(empty) = 0.
sum(leaf(N)) = N.
sum(branch(L, R)) = sum(L) + sum(R).

:- func len(my_list(T)) = int.

len([]) = 0.
len([_ | Xs]) = 1 + len(Xs).

:- func list_strange(list(strange)) = int.

list_strange([]) = 0.
list_strange([X | Xs]) = Value + list_strange(Xs) :-
    ( X = foo(Value)
    ; X = bar(_), Value = 100
    ).
