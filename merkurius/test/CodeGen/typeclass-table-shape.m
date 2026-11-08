% A method table's first word says what the instance's typeclass_infos hold,
% for the message copy and the collector (design document, "A table's slots").
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module typeclass_table_shape.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- typeclass base(T) where [ func one(T) = int ].
:- typeclass derived(T) <= base(T) where [ func two(T) = int ].

:- type pair2(A, B) ---> pair2(A, B).

:- instance base(int) where [ one(X) = X ].
:- instance derived(int) where [ two(X) = X + X ].
:- instance base(pair2(A, B)) <= base(A) where [
    one(pair2(X, _)) = one(X) + 1
].
:- instance derived(pair2(A, B)) <= derived(A) where [
    two(pair2(X, _)) = two(X) + 2
].

:- func total(T) = int <= derived(T).
total(X) = one(X) + two(X).

main(!IO) :-
    io.write_int(total(3), !IO),
    io.nl(!IO),
    io.write_int(total(pair2(3, "unused")), !IO),
    io.nl(!IO).

% One superclass, the type's type_info and no others, no constraints.
% MERK-DAG: merk.instance @"merk.instance.typeclass_table_shape.derived(int)" {{.*}} shape [1, 1, 0]
% None of the first; B has a type_info of its own; one constraint.
% MERK-DAG: merk.instance @"merk.instance.typeclass_table_shape.base(typeclass_table_shape.pair2(V0, V1))" {{.*}} shape [0, 2, 1]
% MERK-DAG: merk.instance @"merk.instance.typeclass_table_shape.derived(typeclass_table_shape.pair2(V0, V1))" {{.*}} shape [1, 2, 1]

% The word is 1 | 1 << 16.
% LLVM: @"merk.instance.typeclass_table_shape.derived(int)" = {{.*}}[ptr inttoptr (i64 65537 to ptr),

% OUTPUT:9
% OUTPUT-NEXT:12
