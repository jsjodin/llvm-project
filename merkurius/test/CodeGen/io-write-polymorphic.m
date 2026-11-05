% io.write of values whose type_info is made at run time, with type variables
% among a type's fields: the walker finds each field's type through the type's
% arguments. A type of many constructors shares a tag, and has constants too.
% The output is mmc's for the same program.
%
% RUN: merkc %s -o %t && %t | diff %S/Inputs/io-write-polymorphic.expected -
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | diff %S/Inputs/io-write-polymorphic.expected -
% RUN: merkc -O2 %s -o %t2 && %t2 | diff %S/Inputs/io-write-polymorphic.expected -
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK

:- module io_write_polymorphic.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, list, maybe.

:- type tree(T) ---> leaf ; node(tree(T), T, tree(T)).
:- type two(A, B) ---> two(A, list(B), maybe({A, B})) ; none.
:- type box(T) ---> box(T).
:- type nested(T) ---> nested(list(list(T))).
:- type many
    --->    c0 ; c1
    ;       a1(int) ; a2(int) ; a3(int) ; a4(int) ; a5(int) ; a6(int)
    ;       a7(int) ; a8(int) ; a9(string, many).

% MERK-DAG: merk.type_ctor "list.list/1" {{.*}} functors {{.*}}"list.list/1", 0
% MERK-DAG: merk.write_word %arg0, %arg1 flags 0

:- pred w(T::in, io::di, io::uo) is det.
w(X, !IO) :-
    io.write(X, !IO),
    io.nl(!IO).

:- pred around(T::in, io::di, io::uo) is det.
around(X, !IO) :-
    w([X, X], !IO),
    w({X, "s"}, !IO),
    w(box(X), !IO),
    w(yes(X), !IO),
    w(node(leaf, X, leaf), !IO),
    w(two(X, [1], yes({X, 1})), !IO),
    w(nested([[X]]), !IO),
    array.init(1, X, A),
    w(A, !IO).

main(!IO) :-
    around(1, !IO),
    around("s", !IO),
    around([1.5], !IO),
    around(box(a9("x", c1)), !IO),
    w([c0, c1, a1(1), a6(6), a7(7), a8(8), a9("n", a8(0))], !IO),
    w(none : two(int, int), !IO),
    w([] : list(int), !IO),
    % A list of a type nothing determines.
    w([], !IO).
