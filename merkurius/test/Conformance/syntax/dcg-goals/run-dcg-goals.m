% MANUAL: DCG-goals
% Each form of DCG goal: some and all, sequence, disjunction, braces, an
% input match, [], both negations, both if-then-else syntaxes, =(Term),
% :=(Term), field selection and update, and calls of a name, call/N and a
% variable.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:['a']
% OUTPUT-NEXT:['w']
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:{['q'], []}
% OUTPUT-NEXT:['c']
% OUTPUT-NEXT:{4, st(4, "set")}
% OUTPUT-NOT:{{.}}

:- module run_dcg_goals.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- import_module char.
:- type st ---> st(n :: int, s :: string).
:- pred digits(list(char)::in, list(char)::out) is det.
digits --> ( if [C], { char.is_digit(C) } then digits else [] ).
:- pred notx(list(char)::in, list(char)::out) is semidet.
notx --> \+ ['x'], not ['y'], [_].
:- pred anyb(list(char)::in, list(char)::out) is semidet.
anyb --> some [C] ( [C], { C = 'b' } ).
:- pred allb(list(char)::in, list(char)::out) is semidet.
allb --> all [C] { list.member(C, ['b']) => C = 'b' }.
:- pred rest(list(char)::out, list(char)::in, list(char)::out) is det.
rest(R) --> =(R), :=([]).
:- pred twice(pred(list(char), list(char)), list(char), list(char)).
:- mode twice(pred(in, out) is semidet, in, out) is semidet.
twice(P) --> P, call(P).
:- pred ab(list(char)::in, list(char)::out) is semidet.
ab --> ( ['a'] ; ['b'] ).
:- pred upd(int::out, st::in, st::out) is det.
upd(N) --> N =^ n, ^ s := "set".

main(!IO) :-
    digits(['1', '2', 'a'], R1), io.write(R1, !IO), io.nl(!IO),
    ( if notx(['z', 'w'], R2) then io.write(R2, !IO) else io.write_string("no", !IO) ), io.nl(!IO),
    ( if anyb(['b'], R3) then io.write(R3, !IO) else io.write_string("no", !IO) ), io.nl(!IO),
    ( if allb([], R7) then io.write(R7, !IO) else io.write_string("no", !IO) ), io.nl(!IO),
    rest(R4, ['q'], R5), io.write({R4, R5}, !IO), io.nl(!IO),
    ( if twice(ab, ['a', 'b', 'c'], R6) then io.write(R6, !IO) else io.write_string("no", !IO) ), io.nl(!IO),
    upd(N, st(4, ""), S), io.write({N, S}, !IO), io.nl(!IO).
