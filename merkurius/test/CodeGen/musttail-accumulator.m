% LLVM's tail recursion elimination turns f's recursive calls, whose
% results are combined after them, into a loop with an accumulator, which
% rewrites each of f's returns to combine it. One return follows a musttail
% call, of truncate_to_int, and must stay next to it (design document,
% "Calling convention"), so the pass keeps the recursive calls; it used to
% rewrite that return too, and -O2 failed with "failed to perform tail call
% elimination on a call site marked musttail". The output is mmc's.
%
% RUN: merkc -O2 %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.O0 && %t.O0 | FileCheck %s --match-full-lines --strict-whitespace

:- module musttail_accumulator.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.

:- func f(int) = int.
f(N) =
    ( if N = 0 then
        truncate_to_int(float(N) + 2.5)
    else if N mod 2 = 0 then
        2 * f(N - 1)
    else
        N + f(N - 1)
    ).

main(!IO) :-
    io.write_int(f(10), !IO),
    io.nl(!IO).
% CHECK:230
