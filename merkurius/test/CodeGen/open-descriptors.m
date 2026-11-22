% The descriptors' marks for the open layout (runtime/open.h;
% docs/INPROGRESS.md, Open types, step 2): kOpenSame (2) on a type whose
% open layout is its ordinary one if its parameters' are, and kClosed (4) on
% closed(T)'s, whose open layout is a box. tree, a tuple, wrap(T) and a type
% of seven constructors with arguments keep their layouts; itree's int field
% is boxed, an enum's constants are shifted, and a type of eight
% constructors with arguments gives up tag 7 to variables, so theirs change.
% Outside open(_), closed(T) is a notag type: T's word, which io.write and a
% deconstruction see as usual.
%
% RUN: merkc -fc1 -fopen-types -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc -fc1 -fopen-types -emit-llvm -triple x86_64-unknown-linux-gnu %S/../../library/open.m | FileCheck %s --check-prefix=LIBRARY
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module open_descriptors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, open.

:- type tree ---> leaf ; node(tree, tree).
:- type itree ---> ileaf ; inode(itree, int, itree).
:- type colour ---> red ; green.
:- type wrap(T) ---> wrap(T).
:- type seven ---> s1(seven) ; s2(seven) ; s3(seven) ; s4(seven) ; s5(seven)
    ; s6(seven) ; s7(tree).
:- type eight ---> e1(eight) ; e2(eight) ; e3(eight) ; e4(eight) ; e5(eight)
    ; e6(eight) ; e7(eight) ; e8(tree).

% LLVM-DAG: @"merk.type_ctor.open_descriptors.tree/0" = {{.*}} { i8 18, i8 2,
% LLVM-DAG: @"merk.type_ctor.open_descriptors.itree/0" = {{.*}} { i8 18, i8 0,
% LLVM-DAG: @"merk.type_ctor.open_descriptors.colour/0" = {{.*}} { i8 16, i8 1,
% LLVM-DAG: @"merk.type_ctor.open_descriptors.wrap/1" = {{.*}} { i8 17, i8 2,
% LLVM-DAG: @"merk.type_ctor.open_descriptors.seven/0" = {{.*}} { i8 18, i8 2,
% LLVM-DAG: @"merk.type_ctor.open_descriptors.eight/0" = {{.*}} { i8 18, i8 0,
% LLVM-DAG: @"merk.type_ctor.{}/2" = {{.*}} { i8 19, i8 2,
% LLVM-DAG: @"merk.type_ctor.int/0" = {{.*}} { i8 0, i8 0,
% closed(T)'s and open(T)'s are the open module's, open(T)'s of kind Open (21).
% LLVM-DAG: @"merk.type_ctor.open.closed/1" = external constant
% LIBRARY-DAG: @"merk.type_ctor.open.closed/1" = constant {{.*}} { i8 17, i8 4,
% LIBRARY-DAG: @"merk.type_ctor.open.open/1" = constant {{.*}} { i8 21, i8 0,

main(!IO) :-
    io.write(node(leaf, leaf), !IO), io.nl(!IO),
    io.write(inode(ileaf, 1, ileaf), !IO), io.nl(!IO),
    io.write([red], !IO), io.nl(!IO),
    io.write(wrap(leaf), !IO), io.nl(!IO),
    io.write({leaf, 1}, !IO), io.nl(!IO),
    io.write(s1(s7(leaf)), !IO), io.nl(!IO),
    io.write(e1(e8(leaf)), !IO), io.nl(!IO),
    C = closed([1, 2]),
    io.write(C, !IO), io.nl(!IO),
    closed(L) = C,
    io.write(L, !IO), io.nl(!IO).
% OUTPUT:node(leaf, leaf)
% OUTPUT-NEXT:inode(ileaf, 1, ileaf)
% OUTPUT-NEXT:[red]
% OUTPUT-NEXT:wrap(leaf)
% OUTPUT-NEXT:{leaf, 1}
% OUTPUT-NEXT:s1(s7(leaf))
% OUTPUT-NEXT:e1(e8(leaf))
% OUTPUT-NEXT:closed([1, 2])
% OUTPUT-NEXT:[1, 2]
