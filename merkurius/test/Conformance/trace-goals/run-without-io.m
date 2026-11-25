% MANUAL: Trace goals
% A trace goal without io, its local variables, and the variables of the
% surrounding code that it reads.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:long 3 6
% OUTPUT-NEXT:6
% OUTPUT-NOT:{{.}}

:- module run_without_io.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(list(int)::in, int::out) is det.
p(L, S) :-
    S = list.foldl(plus, L, 0),
    trace [io(!IO)] (
        Len = list.length(L),
        ( if Len > 2 then io.format("long %d %d\n", [i(Len), i(S)], !IO) else true )
    ),
    trace [] ( X = S + 1, _ = X ).
main(!IO) :- p([1, 2, 3], S), io.write_int(S, !IO), io.nl(!IO).
