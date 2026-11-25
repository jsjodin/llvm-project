% Lists in lockstep in a loop (design document, "Loops", Meaning): a list
% bound before the loop that is at its end too early or too late, beside
% another list or a for, is a software error that names the loop, which a
% try catches, rather than a failure, so that the loop is det. Its tests
% are merk.throw_if, not merk.guard; a for's end is still tested.
%
% RUN: merkc -fc1 -fdo-loops -emit-merk %s -o - | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 -fdo-loops %s -o %t0 && not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2 && not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% MERK-LABEL: merk.proc private @loop_mismatched_lengths.dot__loop_
% MERK-NOT:     merk.guard
% MERK:         merk.throw_if %{{.*}}, "Software Error: mismatched list lengths in the loop at loop-mismatched-lengths.m:[[#@LINE+13]]"
% MERK-NOT:     merk.guard
% MERK-LABEL: merk.proc private @loop_mismatched_lengths.counted__loop_

:- module loop_mismatched_lengths.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string, univ.

:- pred dot(list(int)::in, list(int)::in, int::out) is det.
dot(Xs, Ys, S) :-
    ( foreach(X, Xs), foreach(Y, Ys), fromto(0, S0, S1, S) do S1 = S0 + X * Y ).

:- pred counted(list(int)::in, list(int)::out) is det.
counted(Xs, Zs) :-
    ( for(I, 1, 3), foreach(X, Xs), foreach(Z, Zs) do Z = I * X ).

:- pred try_dot(list(int)::in, list(int)::in, io::di, io::uo) is cc_multi.
try_dot(Xs, Ys, !IO) :-
    ( try [io(!IO)] dot(Xs, Ys, S) then
        io.write_int(S, !IO), io.nl(!IO)
    catch_any E ->
        io.write(E, !IO), io.nl(!IO)
    ).

main(!IO) :-
    try_dot([1, 2, 3], [4, 5, 6], !IO),
    try_dot([1, 2, 3], [4, 5], !IO),
    try_dot([1], [4, 5], !IO),
    counted([1, 2, 3], Z), io.write(Z, !IO), io.nl(!IO),
    counted([1, 2], _).
% OUTPUT:32
% OUTPUT-NEXT:software_error("mismatched list lengths in the loop at loop-mismatched-lengths.m:26")
% OUTPUT-NEXT:software_error("mismatched list lengths in the loop at loop-mismatched-lengths.m:26")
% OUTPUT-NEXT:[1, 4, 9]
% OUTPUT-NEXT:Uncaught Merkurius exception:
% OUTPUT-NEXT:Software Error: mismatched list lengths in the loop at loop-mismatched-lengths.m:30
