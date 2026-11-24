% Under the strict sequential semantics (reference manual, "Semantics"), a
% det call that never returns keeps the program from going on, even when
% nothing uses its output: neither merkc nor LLVM may remove it, as LLVM may
% remove a loop that has no side effects in a function it may assume makes
% progress. Each case loops for ever and is stopped by timeout; were the call
% removed, the program would print and exit 0. mmc also loops.
%
% RUN: merkc -O0 %s -o %t0
% RUN: merkc -O2 %s -o %t2
% RUN: merkc -O3 %s -o %t3
% RUN: not timeout 1 %t0 pred | count 0
% RUN: not timeout 1 %t0 func | count 0
% RUN: not timeout 1 %t2 pred | count 0
% RUN: not timeout 1 %t2 func | count 0
% RUN: not timeout 1 %t3 pred | count 0
% RUN: not timeout 1 %t3 func | count 0
% RUN: not timeout 1 %t3 nested | count 0

:- module strict_infinite_loops.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred loop(int::in, int::out) is det.
loop(N, R) :- loop(N + 1, R).

:- func floop(int) = int.
floop(N) = floop(N + 1).

% The loop is in a callee of the call whose output is unused.
:- pred outer(int::in, int::out) is det.
outer(N, R) :- loop(N, R0), R = R0 + 1.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = ["pred"] then
        loop(0, _)
    else if Args = ["func"] then
        _ = floop(0)
    else
        outer(0, _)
    ),
    io.write_string("the loop was removed\n", !IO).
