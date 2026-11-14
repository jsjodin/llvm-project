% A word of an integer's bits, which a closure's call takes, that the
% optimizer keeps live past safepoints whose bundles do not list it: LICM
% hoists the word of X out of the loop, so one word serves both calls of P
% and every allocation between them. In a spawn, GVN merged the words of
% integers passed to two kinds in the same way (benchmarks/pipeline). That
% stopped the compile with "a root ... has no type at a safepoint". Such a
% word is no pointer (Lowering, OptimizeLLVMIR; Statepoints, IsScalarWord),
% so the collector ignores it; the loop collects at every allocation, and
% checks the stack's roots, while it is live.
%
% mmc gives the same output.
%
% Inputs/gc_scalar_word_spawn.m spawns two GPU process kinds with the same
% integers, as the host compiles it.
% RUN: merkc -fc1 -O2 -fprocess-declarations -S %S/Inputs/gc_scalar_word_spawn.m -o /dev/null
% RUN: merkc -O2 %s -o %t2
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t | FileCheck %s --match-full-lines --strict-whitespace

:- module gc_scalar_word.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    get(41, X, !IO),
    get(0, Z, !IO),
    P = ( if Z > 1 then add1 else add2 ),
    loop(P, 300, X, 0, S, !IO),
    io.format("%d\n", [i(S)], !IO).
% CHECK:71550

:- pred loop(pred(int, int, io, io)::in(pred(in, out, di, uo) is det),
    int::in, int::in, int::in, int::out, io::di, io::uo) is det.
:- pragma no_inline(loop/7).
loop(P, N, X, !S, !IO) :-
    ( if N = 0 then
        true
    else
        P(X, A, !IO),
        L = list.duplicate(N, A),
        P(X, B, !IO),
        !:S = !.S + A + B + list.length(L),
        loop(P, N - 1, X, !S, !IO)
    ).

:- pred add1(int::in, int::out, io::di, io::uo) is det.
add1(X, X + 1, !IO).
:- pred add2(int::in, int::out, io::di, io::uo) is det.
add2(X, X + 2, !IO).

:- pred get(int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C", get(X::in, Y::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure], "Y = X + 1;").
