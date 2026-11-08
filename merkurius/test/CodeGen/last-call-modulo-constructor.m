% Last call modulo constructor (design document, Lowering, "Deterministic and
% semidet code", "Last call modulo constructor"): a procedure makes the cell
% that a recursive call's result goes into first, with a hole for it, and
% calls a variant of the callee that stores the result in the hole, so that
% in the variant the call is a tail call. The programs run on a million
% elements in constant stack, as mmc's do with
% --optimize-constructor-last-call, except odd_pos, shows and double, which
% mmc runs only with a bigger stack. The output is mmc's.
%
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module last_call_modulo_constructor.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

% The recursive call is last modulo the construction of the cell its result
% goes into. The procedure makes the cell, with the hole null, and calls the
% variant, which stores the cell it makes in its caller's hole and calls
% itself last.
% LLVM-LABEL: define internal tailcc ptr addrspace(1) @last_call_modulo_constructor.app_3_0(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) {
% LLVM: store ptr addrspace(1) null, ptr addrspace(1) %{{[0-9]+}}, align 8
% LLVM-NEXT: %[[TERM:[0-9]+]] = getelementptr i8, ptr addrspace(1) %{{[0-9]+}}, i32 1
% LLVM: {{^}}  call tailcc void @last_call_modulo_constructor.app_3_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %2, ptr addrspace(1) %[[TERM]], i64 7)
% LLVM-NEXT: ret ptr addrspace(1) %[[TERM]]
% LLVM-LABEL: define internal tailcc void @last_call_modulo_constructor.app_3_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, i64 %4) {
% LLVM: store ptr addrspace(1) %2, ptr addrspace(1) %{{[0-9]+}}, align 8
% LLVM-NEXT: ret void
% LLVM: %[[CELL:[0-9]+]] = phi ptr addrspace(1)
% LLVM-NEXT: store ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %[[CELL]], align 8
% LLVM-NEXT: %[[HOLE:[0-9]+]] = getelementptr i8, ptr addrspace(1) %[[CELL]], i32 8
% LLVM-NEXT: store ptr addrspace(1) null, ptr addrspace(1) %[[HOLE]], align 8
% LLVM-NEXT: %[[TAGGED:[0-9]+]] = getelementptr i8, ptr addrspace(1) %[[CELL]], i32 1
% LLVM-NEXT: getelementptr i8, ptr addrspace(1) %{{[0-9]+}}, i64 %{{[0-9]+}}
% LLVM-NEXT: store ptr addrspace(1) %[[TAGGED]], ptr addrspace(1) %{{[0-9]+}}, align 8
% LLVM-NEXT: musttail call tailcc void @last_call_modulo_constructor.app_3_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %2, ptr addrspace(1) %[[TAGGED]], i64 7)
% LLVM-NEXT: ret void
:- pred app(list(T)::in, list(T)::in, list(T)::out) is det.
app([], Ys, Ys).
app([X | Xs], Ys, [X | Zs]) :- app(Xs, Ys, Zs).

% In both branches of an if-then-else.
% LLVM-LABEL: define internal tailcc void @last_call_modulo_constructor.mrg_3_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i64 %3) {
% LLVM: musttail call tailcc void @last_call_modulo_constructor.mrg_3_0.lcmc(
% LLVM: musttail call tailcc void @last_call_modulo_constructor.mrg_3_0.lcmc(
:- pred mrg(list(int)::in, list(int)::in, list(int)::out) is det.
mrg([], L, L).
mrg([X | Xs], L2, L) :-
    (
        L2 = [],
        L = [X | Xs]
    ;
        L2 = [Y | Ys],
        ( if X > Y then
            mrg([X | Xs], Ys, L1),
            L = [Y | L1]
        else
            mrg(Xs, [Y | Ys], L1),
            L = [X | L1]
        )
    ).

% A call that is the output itself, in the else branch, is a tail call in
% the variant too.
% LLVM-LABEL: define internal tailcc void @last_call_modulo_constructor.evens_2_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2) {
% LLVM: store ptr addrspace(1) null, ptr addrspace(1) %{{[0-9]+}}, align 8
% LLVM: store ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %{{[0-9]+}}, align 8
% LLVM-NEXT: musttail call tailcc void @last_call_modulo_constructor.evens_2_0.lcmc(ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %{{[0-9]+}}, i64 7)
% LLVM: musttail call tailcc void @last_call_modulo_constructor.evens_2_0.lcmc(ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %1, i64 %2)
:- pred evens(list(int)::in, list(int)::out) is det.
evens([], []).
evens([X | Xs], Es) :-
    ( if X mod 2 = 0 then
        evens(Xs, Es1),
        Es = [X | Es1]
    else
        evens(Xs, Es)
    ).

% Two outputs that go into cells.
% LLVM-LABEL: define internal tailcc void @last_call_modulo_constructor.unzip_3_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, ptr addrspace(1) %3, i64 %4) {
% LLVM: store ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %{{[0-9]+}}, align 8
% LLVM: store ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %{{[0-9]+}}, align 8
% LLVM: musttail call tailcc void @last_call_modulo_constructor.unzip_3_0.lcmc(ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %{{[0-9]+}}, i64 7, ptr addrspace(1) %{{[0-9]+}}, i64 7)
:- pred unzip(list({int, int})::in, list(int)::out, list(int)::out) is det.
unzip([], [], []).
unzip([{A, B} | Ps], [A | As], [B | Bs]) :- unzip(Ps, As, Bs).

% One output that goes into a cell and an accumulator, which the variant
% returns.
% LLVM-LABEL: define internal tailcc { ptr addrspace(1), i64 } @last_call_modulo_constructor.number_4_0(ptr addrspace(1) %0, i64 %1) {
% LLVM: {{^}}  %{{[0-9]+}} = call tailcc i64 @last_call_modulo_constructor.number_4_0.lcmc(
% LLVM-LABEL: define internal tailcc i64 @last_call_modulo_constructor.number_4_0.lcmc(ptr addrspace(1) %0, i64 %1, ptr addrspace(1) %2, i64 %3) {
% LLVM: %[[N:[0-9]+]] = musttail call tailcc i64 @last_call_modulo_constructor.number_4_0.lcmc(
% LLVM-NEXT: ret i64 %[[N]]
:- pred number(list(int)::in, list({int, int})::out, int::in, int::out) is det.
number([], [], !N).
number([X | Xs], [{!.N, X} | Ps], !N) :- !:N = !.N + 1, number(Xs, Ps, !N).

% Mutual recursion.
% Mutual recursion: even_pos's call is the output itself, a tail call to
% odd_pos's variant in even_pos's, which is checked last, as even_pos comes
% after the procedures main calls.
% LLVM-LABEL: define internal tailcc void @last_call_modulo_constructor.odd_pos_2_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2) {
% LLVM: musttail call tailcc void @last_call_modulo_constructor.even_pos_2_0.lcmc(
:- pred odd_pos(list(int)::in, list(int)::out) is det.
odd_pos([], []).
odd_pos([X | Xs], [X | Ys]) :- even_pos(Xs, Ys).
:- pred even_pos(list(int)::in, list(int)::out) is det.
even_pos([], []).
even_pos([_ | Xs], Ys) :- odd_pos(Xs, Ys).

% A semidet procedure.
% LLVM-LABEL: define internal tailcc i1 @last_call_modulo_constructor.upto_3_0.lcmc(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i64 %3) {
% LLVM: %[[OK:[0-9]+]] = musttail call tailcc i1 @last_call_modulo_constructor.upto_3_0.lcmc(
% LLVM-NEXT: ret i1 %[[OK]]
:- pred upto(int::in, list(int)::in, list(int)::out) is semidet.
upto(N, L, Start) :-
    ( if N = 0 then
        Start = []
    else
        L = [X | Xs],
        upto(N - 1, Xs, Start1),
        Start = [X | Start1]
    ).

% Of two recursive calls, only the last one.
% LLVM-LABEL: define internal tailcc void @last_call_modulo_constructor.double_f1_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2) {
% LLVM: {{^}}  %{{[0-9]+}} = call tailcc ptr addrspace(1) @last_call_modulo_constructor.double_f1_0(
% LLVM: musttail call tailcc void @last_call_modulo_constructor.double_f1_0.lcmc(
:- type tree ---> leaf ; node(tree, int, tree).
:- func double(tree) = tree.
double(leaf) = leaf.
double(node(L, X, R)) = node(double(L), 2 * X, double(R)).

% An existentially typed constructor's hidden field comes first in its cell.
% LLVM-LABEL: define internal tailcc void @last_call_modulo_constructor.shows_f1_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2) {
% LLVM: %[[CELL:[0-9]+]] = phi ptr addrspace(1)
% LLVM-NEXT: store ptr addrspace(1) addrspacecast (ptr @merk.typeclass_info.{{[0-9]+}} to ptr addrspace(1)), ptr addrspace(1) %[[CELL]], align 8
% LLVM-NEXT: %[[X:[0-9]+]] = getelementptr i8, ptr addrspace(1) %[[CELL]], i32 8
% LLVM-NEXT: store ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %[[X]], align 8
% LLVM-NEXT: %[[HOLE:[0-9]+]] = getelementptr i8, ptr addrspace(1) %[[CELL]], i32 16
% LLVM-NEXT: store ptr addrspace(1) null, ptr addrspace(1) %[[HOLE]], align 8
% LLVM-NEXT: %[[TAGGED:[0-9]+]] = getelementptr i8, ptr addrspace(1) %[[CELL]], i32 1
% LLVM-NEXT: getelementptr i8, ptr addrspace(1) %{{[0-9]+}}, i64 %{{[0-9]+}}
% LLVM-NEXT: store ptr addrspace(1) %[[TAGGED]], ptr addrspace(1) %{{[0-9]+}}, align 8
% LLVM-NEXT: musttail call tailcc void @last_call_modulo_constructor.shows_f1_0.lcmc(ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %[[TAGGED]], i64 15)
% LLVM-LABEL: define internal tailcc void @last_call_modulo_constructor.even_pos_2_0.lcmc(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2) {
% LLVM: musttail call tailcc void @last_call_modulo_constructor.odd_pos_2_0.lcmc(ptr addrspace(1) %{{[0-9]+}}, ptr addrspace(1) %1, i64 %2)
:- type shown ---> some [T] shown(T, shown) => show(T) ; none.
:- typeclass show(T) where [ func show(T) = int ].
:- instance show(int) where [ show(X) = 10 * X ].
:- func shows(list(int)) = shown.
shows([]) = none.
shows([X | Xs]) = 'new shown'(X, shows(Xs)).
:- func strings(shown) = list(int).
strings(none) = [].
strings(shown(X, S)) = [show(X) | strings(S)].

:- func spine(int, tree) = tree.
spine(N, T) = ( if N = 0 then T else spine(N - 1, node(leaf, N, T)) ).
:- func right(tree, int) = int.
right(leaf, S) = S.
right(node(_, X, R), S) = right(R, S + X).

:- func count(int) = list(int).
count(N) = L :- count(N, [], L).
:- pred count(int::in, list(int)::in, list(int)::out) is det.
count(N, !L) :- ( if N = 0 then true else !:L = [N | !.L], count(N - 1, !L) ).

main(!IO) :-
    app([1, 2], [3], L1), io.write(L1, !IO), io.nl(!IO),
    mrg([1, 4, 6], [2, 3, 7], L2), io.write(L2, !IO), io.nl(!IO),
    evens([1, 2, 3, 4], L3), io.write(L3, !IO), io.nl(!IO),
    unzip([{1, 2}, {3, 4}], As, Bs), io.write({As, Bs}, !IO), io.nl(!IO),
    number([5, 6, 7], Ps, 10, N), io.write({Ps, N}, !IO), io.nl(!IO),
    odd_pos([1, 2, 3, 4, 5], L4), io.write(L4, !IO), io.nl(!IO),
    ( if upto(2, [1, 2, 3], L5) then io.write(L5, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    ( if upto(4, [1, 2, 3], L6) then io.write(L6, !IO) else io.write_string("no", !IO) ),
    io.nl(!IO),
    io.write(double(node(node(leaf, 1, leaf), 2, node(leaf, 3, leaf))), !IO),
    io.nl(!IO),
    io.write(strings(shows([1, 2, 3])), !IO), io.nl(!IO),
    % A million elements, in constant stack.
    Big = count(1000000),
    app(Big, [0], Big1), io.write_int(list.det_last(Big1), !IO), io.nl(!IO),
    mrg(Big, Big, Big2), io.write_int(list.length(Big2), !IO), io.nl(!IO),
    evens(Big, Big3), io.write_int(list.det_last(Big3), !IO), io.nl(!IO),
    unzip(list.map((func(X) = {X, X + 1}), Big), Big4, Big5),
    io.write_int(list.det_last(Big4) + list.det_last(Big5), !IO), io.nl(!IO),
    number(Big, Big6, 0, Big7), io.write_int(list.length(Big6) + Big7, !IO),
    io.nl(!IO),
    odd_pos(Big, Big8), io.write_int(list.length(Big8), !IO), io.nl(!IO),
    ( if upto(999999, Big, Big9) then
        io.write_int(list.det_last(Big9), !IO)
    else
        io.write_string("no", !IO)
    ),
    io.nl(!IO),
    io.write_int(list.length(strings(shows(Big))), !IO), io.nl(!IO),
    io.write_int(right(double(spine(1000000, leaf)), 0), !IO), io.nl(!IO).

% OUTPUT:[1, 2, 3]
% OUTPUT-NEXT:[1, 2, 3, 4, 6, 7]
% OUTPUT-NEXT:[2, 4]
% OUTPUT-NEXT:{[1, 3], [2, 4]}
% OUTPUT-NEXT:{[{10, 5}, {11, 6}, {12, 7}], 13}
% OUTPUT-NEXT:[1, 3, 5]
% OUTPUT-NEXT:[1, 2]
% OUTPUT-NEXT:no
% OUTPUT-NEXT:node(node(leaf, 2, leaf), 4, node(leaf, 6, leaf))
% OUTPUT-NEXT:[10, 20, 30]
% OUTPUT-NEXT:0
% OUTPUT-NEXT:2000000
% OUTPUT-NEXT:1000000
% OUTPUT-NEXT:2000001
% OUTPUT-NEXT:2000000
% OUTPUT-NEXT:500000
% OUTPUT-NEXT:999999
% OUTPUT-NEXT:1000000
% OUTPUT-NEXT:1000001000000
