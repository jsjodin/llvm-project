% RUN: merkc -O0 %s -o %t0
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t0.O2
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% A string compared in a loop made from tail recursion, across collections:
% the C pointer that strcmp takes must be the string's address after each one.
:- module gc_c_pointer.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module int, list, string.

:- type tree
    --->    leaf
    ;       node(tree, string, tree).

:- func insert(string, tree) = tree.
insert(K, leaf) = node(leaf, K, leaf).
insert(K, node(L, K0, R)) = Result :-
    compare(Order, K, K0),
    (
        Order = (<),
        Result = node(insert(K, L), K0, R)
    ;
        Order = (>),
        Result = node(L, K0, insert(K, R))
    ;
        Order = (=),
        Result = node(L, K, R)
    ).

:- func flatten(tree) = list(string).
flatten(leaf) = [].
flatten(node(L, K, R)) = flatten(L) ++ [K | flatten(R)].

:- func key(int) = string.
key(I) = "k" ++ string.int_to_string((I * 37) mod 101).

main(!IO) :-
    Keys = list.map(key, 1 `..` 40),
    Tree = list.foldl(insert, Keys, leaf),
    io.write_string(string.join_list(" ", flatten(Tree)), !IO),
    io.nl(!IO).

% OUTPUT:k10 k100 k13 k16 k19 k20 k23 k26 k29 k3 k30 k33 k36 k37 k40 k43 k46 k47 k50 k53 k56 k57 k6 k60 k63 k66 k67 k70 k73 k74 k77 k80 k83 k84 k87 k9 k90 k93 k94 k97
