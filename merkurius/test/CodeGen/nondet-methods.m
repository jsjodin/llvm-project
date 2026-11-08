% Nondet and multi methods: a call through the typeclass_info is
% merk.nondet_call_method, which outlining makes a semidet merk.call_method
% with a continuation, and the instance's dispatch procedure passes the
% continuation on to the instance's procedure, as a tail call. A call of a
% known instance's method calls its procedure with merk.nondet_call. The
% output is mmc's for the same program.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module nondet_methods.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list, solutions.

:- typeclass gen(T) where [
    pred elem(T::in, int::out) is nondet,
    pred several(T::in, int::out) is multi,
    pred pick(T::in, U::in, U::out) is multi
].

:- type three ---> three.
:- type range ---> range(int, int).

:- instance gen(three) where [
    (elem(three, X) :- ( X = 1 ; X = 2 ; X = 3 )),
    (several(three, X) :- ( X = 10 ; X = 20 )),
    (pick(three, U, V) :- ( V = U ; V = U ))
].

:- instance gen(range) where [
    pred(elem/2) is range_elem,
    (several(range(L, _), X) :- ( X = L ; X = L + 100 )),
    (pick(range(_, _), U, V) :- V = U)
].

:- pred range_elem(range::in, int::out) is nondet.
range_elem(range(L, H), X) :-
    L =< H,
    ( X = L ; range_elem(range(L + 1, H), X) ).

% Calls through the typeclass_info.
:- pred all_elems(T::in, list(int)::out) is det <= gen(T).
all_elems(G, Xs) :- solutions((pred(X::out) is nondet :- elem(G, X)), Xs).

:- pred sum_several(T::in, int::out) is det <= gen(T).
sum_several(G, S) :-
    promise_equivalent_solutions [S]
    unsorted_aggregate((pred(X::out) is multi :- several(G, X)),
        (pred(X::in, A0::in, A::out) is det :- A = A0 + X), 0, S).

% A method's solutions as a nondet procedure's.
:- pred evens(T::in, int::out) is nondet <= gen(T).
evens(G, X) :- elem(G, X), X mod 2 = 0.

:- pred show(T::in, io::di, io::uo) is det.
show(X, !IO) :- io.write(X, !IO), io.nl(!IO).

main(!IO) :-
    all_elems(three, A), show(A, !IO),
    all_elems(range(4, 9), B), show(B, !IO),
    sum_several(three, C), show(C, !IO),
    sum_several(range(5, 6), D), show(D, !IO),
    solutions(evens(range(1, 10)), E), show(E, !IO),
    % Calls of a known instance's method.
    solutions((pred(Y::out) is multi :- pick(three, "s", Y)), F), show(F, !IO),
    promise_equivalent_solutions [G]
        unsorted_solutions((pred(Y::out) is multi :- pick(three, 'c', Y)), G),
    show(G, !IO),
    ( if elem(range(3, 5), H), H > 3 then show(H, !IO) else show("none", !IO) ),
    ( if elem(three, I), I > 3 then show(I, !IO) else show("none", !IO) ).

% MERK-LABEL: merk.proc @nondet_methods.main_2_0()
% MERK: merk.nondet_call @"nondet_methods.elem_2_0$nondet_methods.gen(nondet_methods.range)"(
% MERK: merk.nondet_call @"nondet_methods.elem_2_0$nondet_methods.gen(nondet_methods.three)"(
% MERK-LABEL: merk.proc private @nondet_methods.evens_2_0$words(%arg0: !merk.typeclass_info, %arg1: !merk.word) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
% MERK: merk.nondet_call_method %arg0[0] (%arg1 : !merk.word) {
% MERK-NEXT: ^bb0(%{{.*}}: i64):
% MERK-LABEL: merk.proc private @nondet_methods.sum_several_2_0$lambda3(%arg0: !merk.type_info, %arg1: !merk.typeclass_info, %arg2: !merk.word) -> !merk.word attributes {determinism = #merk.determinism<multi>} {
% MERK-NEXT: merk.nondet_call_method %arg1[1] (%arg2 : !merk.word) {
% MERK-LABEL: merk.proc private @"nondet_methods.pick_3_0$nondet_methods.gen(nondet_methods.three)$dispatch"(%arg0: !merk.typeclass_info, %arg1: !merk.type_info, %arg2: !merk.word, %arg3: !merk.word) -> !merk.word attributes {determinism = #merk.determinism<multi>} {
% MERK-NEXT: %[[T:.*]] = merk.from_word %arg2 : !merk.term<"nondet_methods.three">
% MERK-NEXT: merk.nondet_call @"nondet_methods.pick_3_0$nondet_methods.gen(nondet_methods.three)"(%arg1, %[[T]], %arg3 : !merk.type_info, !merk.term<"nondet_methods.three">, !merk.word) {
% MERK-NEXT: ^bb0(%[[U:.*]]: !merk.word):
% MERK-NEXT: merk.succeed %[[U]] : !merk.word

% LLVM-LABEL: define internal tailcc i1 @"nondet_methods.evens_2_0$words"(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr %2, ptr %3)
% LLVM: %[[TABLE:.*]] = load ptr, ptr addrspace(1) %0
% LLVM-NEXT: %[[SLOT:.*]] = getelementptr i8, ptr %[[TABLE]], i32 8
% LLVM-NEXT: %[[METHOD:.*]] = load ptr, ptr %[[SLOT]]
% LLVM-NEXT: call tailcc i1 %[[METHOD]](ptr addrspace(1) %0, ptr addrspace(1) %1, ptr @"nondet_methods.evens_2_0$words.cont.0", ptr %{{.*}})
% LLVM-LABEL: define internal tailcc i1 @"nondet_methods.elem_2_0$nondet_methods.gen(nondet_methods.three)$dispatch"(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr %2, ptr %3)
% LLVM: musttail call tailcc i1 @"nondet_methods.elem_2_0$nondet_methods.gen(nondet_methods.three)"(i64 %{{[0-9]+}}, ptr %2, ptr %3)

% OUTPUT:[1, 2, 3]
% OUTPUT-NEXT:[4, 5, 6, 7, 8, 9]
% OUTPUT-NEXT:30
% OUTPUT-NEXT:110
% OUTPUT-NEXT:[2, 4, 6, 8, 10]
% OUTPUT-NEXT:["s"]
% OUTPUT-NEXT:['c', 'c']
% OUTPUT-NEXT:4
% OUTPUT-NEXT:"none"
