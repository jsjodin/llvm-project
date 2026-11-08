% array/1 and from_list make an array of a list's elements, in order: of
% floats, an empty list, strings, characters, narrow integers, terms and
% arrays, and of a long list made at run time. merk.array_from_list lowers
% to a call of the runtime, which follows the list's cells by the tag of
% [|] in the list's layout. The output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module array_from_list.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, char, int, int8, list.

:- type shape ---> circle(float) ; square(int) ; none.

main(!IO) :-
    Floats = array([1.0, 2.5, 4.0]),
    io.write(Floats, !IO),
    io.nl(!IO),
    Empty = array([]) : array(int),
    io.write(Empty, !IO),
    io.nl(!IO),
    from_list(["one", "two", "three"], Strings),
    io.write(Strings, !IO),
    io.nl(!IO),
    Chars = from_list(['a', 'b', 'c']) : array(char),
    io.write(Chars, !IO),
    io.nl(!IO),
    Narrow = array([-1i8, 2i8, -128i8]),
    io.write(Narrow, !IO),
    io.nl(!IO),
    Shapes = array([circle(1.5), none, square(3)]),
    io.write(Shapes, !IO),
    io.nl(!IO),
    % A list made at run time, long enough that a recursive copy would
    % show, and its array's elements read back.
    Long = array(upto(1, 200000)),
    io.write_int(array.size(Long), !IO),
    io.nl(!IO),
    io.write_int(Long ^ elem(0) + Long ^ elem(199999), !IO),
    io.nl(!IO),
    Nested = array([array([1, 2]), array([3])]),
    io.write(Nested, !IO),
    io.nl(!IO).

:- func upto(int, int) = list(int).

upto(Lo, Hi) = L :-
    upto_acc(Hi, Lo, [], L).

:- pred upto_acc(int::in, int::in, list(int)::in, list(int)::out) is det.

upto_acc(N, Lo, !L) :-
    ( if N < Lo then
        true
    else
        !:L = [N | !.L],
        upto_acc(N - 1, Lo, !L)
    ).

% MERK-LABEL: merk.proc @array_from_list.main_2_0
% MERK: merk.array_from_list %{{.*}} : <"list.list(float)"> -> <f64>
% MERK: merk.array_from_list %{{.*}} : <"list.list(int)"> -> <i64>
% MERK: merk.array_from_list %{{.*}} : <"list.list(string)"> -> <!merk.string>
% MERK: merk.array_from_list %{{.*}} : <"list.list(character)"> -> <i32>
% MERK: merk.array_from_list %{{.*}} : <"list.list(int8)"> -> <i8>
% MERK: merk.array_from_list %{{.*}} : <"list.list(array_from_list.shape)"> -> <!merk.term<"array_from_list.shape">>

% The tag of [|] is 1, after the constants' 0; [] is the word 0.
% LLVM-LABEL: define {{.*}} @array_from_list.main_2_0(
% LLVM: call ptr addrspace(1) @merk_array_from_list(ptr addrspace(1) getelementptr (i8, ptr addrspace(1) addrspacecast (ptr @merk.term.{{[0-9]+}} to ptr addrspace(1)), i64 1), i64 1)
% LLVM: call ptr addrspace(1) @merk_array_from_list(ptr addrspace(1) null, i64 1)

% OUTPUT:array([1.0, 2.5, 4.0])
% OUTPUT-NEXT:array([])
% OUTPUT-NEXT:array(["one", "two", "three"])
% OUTPUT-NEXT:array(['a', 'b', 'c'])
% OUTPUT-NEXT:array([-1i8, 2i8, -128i8])
% OUTPUT-NEXT:array([circle(1.5), none, square(3)])
% OUTPUT-NEXT:200000
% OUTPUT-NEXT:200001
% OUTPUT-NEXT:array([array([1, 2]), array([3])])
% OUTPUT-EMPTY:
