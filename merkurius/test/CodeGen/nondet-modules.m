% Nondet and multi procedures of another module: outlining gives the
% declarations the procedures' new types, so that the calls agree with the
% definitions, and a call in tail position of a multi procedure passes on
% its continuation. A polymorphic callee gives a word, which the
% continuation converts, so that call is not one. The output is mmc's.
%
% RUN: merkc -I %S/Inputs %S/Inputs/ndgive.m %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu -I %S/Inputs %s -o - | FileCheck %s --check-prefix=LLVM

:- module nondet_modules.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list, ndgive.

:- pred big(int::out, list(int)::in) is nondet.
big(X, L) :-
    L = [_, _ | _],
    elem(X, L).

:- pred first_big(list(int)::in, int::out) is cc_nondet.
first_big(L, X) :- big(X, L), X > 3.

:- pred also_small(int::out) is multi.
also_small(X) :- small(X).

:- pred first_sum(int::in, int::out, int::out) is cc_nondet.
first_sum(S, X, Y) :- also_small(X), small(Y), X + Y = S.

:- pred first_name(list(string)::in, string::out) is cc_nondet.
first_name(L, X) :- elem(X, L), X \= "a".

main(!IO) :-
    ( if first_big([1, 5, 2, 7], B) then io.write_int(B, !IO) else io.write_string("none", !IO) ),
    io.nl(!IO),
    ( if first_sum(5, X, Y) then io.write_int(X, !IO), io.write_string(" ", !IO), io.write_int(Y, !IO) else io.write_string("none", !IO) ),
    io.nl(!IO),
    ( if first_name(["a", "b"], N) then io.write_string(N, !IO) else io.write_string("none", !IO) ),
    io.nl(!IO).

% LLVM-LABEL: define internal tailcc i1 @nondet_modules.big_2_0(ptr %0, ptr %1, ptr %2)
% LLVM: call tailcc i1 @ndgive.elem_2_0(ptr @"merk.type_ctor.int/0", ptr %{{.*}}, ptr @nondet_modules.big_2_0.cont.0, ptr %{{.*}})
% LLVM: declare tailcc i1 @ndgive.elem_2_0(ptr, ptr, ptr, ptr)
% LLVM-LABEL: define internal tailcc i1 @nondet_modules.also_small_1_0(ptr %0, ptr %1)
% LLVM: musttail call tailcc i1 @ndgive.small_1_0(ptr %0, ptr %1)

% CHECK:5
% CHECK-NEXT:2 3
% CHECK-NEXT:b
