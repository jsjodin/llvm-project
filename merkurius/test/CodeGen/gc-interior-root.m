% A root that points into its cell rather than at it (runtime/roots.cpp,
% InteriorSlot). At -O2 LLVM computes the address of gather's array's
% elements once, before the loop, and the statepoint rewrite carries it round
% the loop as a pointer derived from the array. The array is one of two, so
% the rewrite makes it a base of its own, which no bundle lists either. The
% collector took the elements' address for a cell and stopped with "an
% update-only root whose cell nothing else reaches". Every allocation
% collects, so the elements must follow their array each time.
%
% RUN: merkc -O2 %s -o %t2
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --match-full-lines --strict-whitespace

:- module gc_interior_root.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string.

main(!IO) :-
    A = array.generate(50, (func(I) = I * 3)),
    B = array.generate(70, (func(I) = I + 1000)),
    rounds(0, 6, A, B, 0, Sum),
    io.format("%d\n", [i(Sum)], !IO).

    % Each round gathers from one array or the other.
:- pred rounds(int::in, int::in, array(int)::in, array(int)::in,
    int::in, int::out) is det.
rounds(R, N, A, B, !Sum) :-
    ( if R < N then
        ( if R mod 2 = 0 then C = A else C = B ),
        gather(C, 0, 200, [], L),
        !:Sum = !.Sum + list.foldl(int.plus, L, 0) + array.size(C),
        rounds(R + 1, N, C, A, !Sum)
    else
        true
    ).

    % Allocates a cell for each element it reads.
:- pred gather(array(int)::in, int::in, int::in,
    list(int)::in, list(int)::out) is det.
gather(C, I, N, !L) :-
    ( if I < N then
        !:L = [C ^ unsafe_elem(I mod 50) | !.L],
        gather(C, I + 1, N, !L)
    else
        true
    ).

% CHECK:88500
