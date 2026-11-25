% A flat record and a float in the environment of nondeterministic code
% (design document, "Environments are in the frame that makes the call",
% and GC metadata): the environment's layout for the collector gives each
% field's byte offset as the data layout places it, a nested record's size
% and alignment from its fields', so that the string after them is found.
% Each allocation collects and checks the roots. The output is mmc's.
%
% RUN: merkc -O0 %s -o %t0 && env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t0 2>&1 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

:- module flat_record_environment.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, float, int, int8, int32, list, solutions, string.

:- type inner ---> inner(int8, int).
:- type rec ---> rec(inner, int32, char, float).

:- func mk(int) = rec.
:- pragma no_inline(mk/1).
mk(N) = rec(inner(int8.cast_from_int(N mod 100), N * 2), int32.cast_from_int(N), 'q', 0.5).

:- func show(rec, int, float) = string.
:- pragma no_inline(show/3).
show(rec(inner(A, B), C, D, E), I, F) = string.format("%d:%d,%d,%d,%c,%.1f,%.2f",
    [i(I), i(int8.to_int(A)), i(B), i(int32.to_int(C)), c(D), f(E), f(F)]).

:- pred gen(int::in, string::out) is nondet.
gen(K, S) :-
    R = mk(K),
    F = float(K) / 4.0,
    T = string.from_char_list(['<', '>']),
    list.member(I, [1, 2, 3]),
    L = list.duplicate(I, "z"),
    S = string.join_list("", L) ++ T ++ show(R, I, F).

main(!IO) :-
    solutions(gen(7), Ss),
    io.write_list(Ss, " ", io.write_string, !IO), io.nl(!IO).
% CHECK:Merkurius runtime: roots: {{[0-9]+}} safepoints, {{[0-9]+}} frames, {{[0-9]+}} roots, {{[1-9][0-9]*}} environments, {{[0-9]+}} slow paths not at a safepoint
% CHECK-NEXT:z<>1:7,14,7,q,0.5,1.75 zz<>2:7,14,7,q,0.5,1.75 zzz<>3:7,14,7,q,0.5,1.75
