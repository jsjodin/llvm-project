% MANUAL: Trace goals
% Trace goals with io in det, semidet and nondet code, in a function, and
% in main beside its own io state, which run when reached and again on
% backtracking.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:d 3
% OUTPUT-NEXT:A=6
% OUTPUT-NEXT:s 1
% OUTPUT-NEXT:not s1
% OUTPUT-NEXT:s 5
% OUTPUT-NEXT:s5
% OUTPUT-NEXT:n 1
% OUTPUT-NEXT:n 2
% OUTPUT-NEXT:n 3
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:n 1
% OUTPUT-NEXT:n 2
% OUTPUT-NEXT:some Z > 1
% OUTPUT-NEXT:f
% OUTPUT-NEXT:11
% OUTPUT-NEXT:in main
% OUTPUT-NEXT:end
% OUTPUT-NOT:{{.}}

:- module run_io.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions, string.
:- pred d(int::in, int::out) is det.
d(X, Y) :- trace [io(!IO)] ( io.format("d %d\n", [i(X)], !IO) ), Y = X * 2.
:- pred s(int::in) is semidet.
s(X) :- trace [io(!IO)] ( io.format("s %d\n", [i(X)], !IO) ), X > 2.
:- pred n(int::out) is multi.
n(X) :- ( X = 1 ; X = 2 ; X = 3 ), trace [io(!S)] ( io.format("n %d\n", [i(X)], !S) ).
:- func f(int) = int.
f(X) = Y :- Y = X + 1, trace [io(!IO)] ( io.write_string("f\n", !IO) ).
main(!IO) :-
    d(3, A), io.format("A=%d\n", [i(A)], !IO),
    ( if s(1) then io.write_string("s1\n", !IO) else io.write_string("not s1\n", !IO) ),
    ( if s(5) then io.write_string("s5\n", !IO) else io.write_string("not s5\n", !IO) ),
    solutions(n, L), io.write(L, !IO), io.nl(!IO),
    ( if n(Z), Z > 1 then io.write_string("some Z > 1\n", !IO) else true ),
    io.write_int(f(10), !IO), io.nl(!IO),
    trace [io(!T)] ( io.write_string("in main\n", !T) ),
    io.write_string("end\n", !IO).
