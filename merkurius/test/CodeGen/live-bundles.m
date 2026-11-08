% Each call of Mercury code and each allocation's slow path has a deopt bundle
% of what is live across it, a type's index before each value
% (docs/INPROGRESS.md, Statepoints and stack maps). A polymorphic root keeps
% the type_info live. A value used only in the other branch is not live.
%
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module live_bundles.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- pragma no_inline(keep/1).
:- func keep(list(T)) = list(T).
keep(L) = L.

% LLVM-LABEL: define internal tailcc ptr addrspace(1) @live_bundles.pa_f2_0(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2)
% LLVM: call tailcc ptr addrspace(1) @live_bundles.keep_f1_0(ptr addrspace(1) %0, ptr addrspace(1) %1) [ "deopt"(i64 {{[0-9]+}}, ptr addrspace(1) %0, i64 {{[0-9]+}}, ptr addrspace(1) %2) ]
% LLVM: call ptr addrspace(1) @merk_alloc_slow(i64 16) [ "deopt"(i64 {{[0-9]+}}, ptr addrspace(1) %2, i64 {{[0-9]+}}, ptr addrspace(1) %{{[0-9]+}}, i64 {{[0-9]+}}, ptr addrspace(1) %0) ]
:- pragma no_inline(pa/2).
:- func pa(list(T), T) = list(T).
pa(L, X) = R :- R0 = keep(L), R = [X | R0].

% What is not used after a call is not live across it, and in a tail call
% nothing is.
% LLVM-LABEL: define internal tailcc ptr addrspace(1) @live_bundles.last_f1_0(ptr addrspace(1) %0)
% LLVM: musttail call tailcc ptr addrspace(1) @live_bundles.twice_f1_0(ptr addrspace(1) %0) [ "deopt"() ]
:- pragma no_inline(last/1).
:- func last(list(int)) = list(int).
last(L) = twice(L).

:- pragma no_inline(twice/1).
:- func twice(list(int)) = list(int).
twice(L) = keep(keep(L)).

% B is used only in the else branch, so the call in the then branch does not
% list it.
% LLVM-LABEL: define internal tailcc ptr addrspace(1) @live_bundles.pick_f3_0(i64 %0, ptr addrspace(1) %1, ptr addrspace(1) %2)
% LLVM: call tailcc ptr addrspace(1) @live_bundles.keep_f1_0(ptr addrspace(1) addrspacecast (ptr @"merk.type_ctor.int/0" to ptr addrspace(1)), ptr addrspace(1) %1) [ "deopt"() ]
:- pragma no_inline(pick/3).
:- func pick(int, list(int), list(int)) = list(int).
pick(N, A, B) = R :-
    ( if N > 0 then
        R = keep(A)
    else
        R = B
    ).

% An environment in the frame is listed by its address, with its layout's
% index, also at a call it is given to.
% LLVM-LABEL: define internal tailcc i1 @live_bundles.pairs_2_0(
% LLVM: call tailcc i1 @live_bundles.each_2_0({{.*}}) [ "deopt"(i64 {{[0-9]+}}, ptr %{{[0-9]+}}) ]
:- pragma no_inline(each/2).
:- pred each(list(int)::in, int::out) is nondet.
each(L, X) :- list.member(X, L).

:- pragma no_inline(pairs/2).
:- pred pairs(list(int)::in, int::out) is nondet.
pairs(L, Y) :- each(L, X), each(L, K), Y = X + K.

main(!IO) :-
    L = pa([1], 2),
    io.write_int(list.length(L), !IO),
    io.nl(!IO),
    io.write_int(list.length(pick(1, last([3]), [])), !IO),
    io.nl(!IO),
    solutions(pairs([1, 2]), Sums),
    io.write_int(list.length(Sums), !IO),
    io.nl(!IO).

% OUTPUT:2
% OUTPUT-NEXT:1
% OUTPUT-NEXT:3
