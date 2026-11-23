% throw, try, try_io and rethrow (docs/INPROGRESS.md, Exceptions, step 1):
% a try is a closure call of the runtime's entry, which keeps a handler in its
% frame, and a throw jumps there. Each mode of try and try_io; a throw from
% under nested calls; a try in a try's goal, which catches, or rethrows to the
% outer; and an uncaught exception, which ends the program in main. With a
% collection at every allocation, the goal allocates before it throws, and
% the caller's roots, which the collector reaches through the entry's frame
% (runtime/roots.cpp, IsTryReturn), are moved under it; after the catch they
% hold what they held. OUTPUT is mmc's, but for the header's "Merkurius".
% RUN: merkc -O0 %s -o %t0
% RUN: not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module exceptions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, solutions, string, univ.

:- type my_error
    --->    my_error(int, string).

:- pred det_goal(int::in, int::out) is det.

det_goal(X, Y) :-
    ( if X > 2 then
        deep(X, 10, Y)
    else
        Y = X * 10
    ).

    % Throws from under N calls, each with a list live across the next.
:- pred deep(int::in, int::in, int::out) is det.

deep(X, N, Y) :-
    L = 1 .. N,
    ( if N = 0 then
        throw(my_error(X, "big"))
    else
        deep(X, N - 1, Y0),
        Y = Y0 + list.foldl(int.plus, L, 0)
    ).

:- pred semi_goal(int::in, int::out) is semidet.

semi_goal(X, Y) :-
    X > 0,
    ( if X > 2 then
        throw("semi " ++ int_to_string(X))
    else
        Y = X
    ).

:- pred multi_goal(int::out) is multi.

multi_goal(1).
multi_goal(2).

:- pred nondet_goal(int::in, int::out) is nondet.

nondet_goal(X, Y) :-
    list.member(Y, [X, X + 1]),
    ( if Y > 5 then
        throw(Y)
    else
        Y > 1
    ).

:- pred cc_multi_goal(int::out) is cc_multi.

cc_multi_goal(Y) :-
    multi_goal(Y).

:- pred cc_nondet_goal(int::in, int::out) is cc_nondet.

cc_nondet_goal(X, Y) :-
    nondet_goal(X, Y).

:- pred io_goal(int::in, int::out, io::di, io::uo) is det.

io_goal(X, Y, !IO) :-
    io.write_string("io_goal\n", !IO),
    ( if X > 2 then
        throw(X)
    else
        Y = X + 1
    ).

:- pred cc_io_goal(int::out, io::di, io::uo) is cc_multi.

cc_io_goal(Y, !IO) :-
    multi_goal(Y),
    io.write_string("cc_io_goal\n", !IO).

:- pred show(exception_result(int)::in, io::di, io::uo) is det.

show(succeeded(V), !IO) :-
    io.format("succeeded(%d)\n", [i(V)], !IO).
show(failed, !IO) :-
    io.write_string("failed\n", !IO).
show(exception(U), !IO) :-
    ( if univ_to_type(U, my_error(N, S)) then
        io.format("my_error %d %s\n", [i(N), s(S)], !IO)
    else if univ_to_type(U, Str) then
        io.format("string %s\n", [s(Str)], !IO)
    else if univ_to_type(U, I) then
        io.format("int %d\n", [i(I)], !IO)
    else
        io.write_string("other\n", !IO)
    ).

    % A lambda that catches its own exception.
:- pred inner_catches(int::out) is cc_multi.

inner_catches(Z) :-
    try(det_goal(4), R),
    ( if R = succeeded(Z0) then
        Z = Z0
    else
        Z = -1
    ).

    % A lambda that rethrows to the outer try.
:- pred inner_rethrows(int::out) is cc_multi.

inner_rethrows(Z) :-
    try(det_goal(4), R),
    ( if R = exception(_) then
        rethrow(R)
    else
        Z = 0
    ).

main(!IO) :-
    Kept = list.map(func(X) = X * X, 1 .. 20),
    % OUTPUT:succeeded(10)
    try(det_goal(1), R1),
    show(R1, !IO),
    % OUTPUT-NEXT:my_error 5 big
    try(det_goal(5), R2),
    show(R2, !IO),
    % OUTPUT-NEXT:failed
    try(semi_goal(0), R3),
    show(R3, !IO),
    % OUTPUT-NEXT:succeeded(2)
    try(semi_goal(2), R4),
    show(R4, !IO),
    % OUTPUT-NEXT:string semi 7
    try(semi_goal(7), R5),
    show(R5, !IO),
    % OUTPUT-NEXT:io_goal
    % OUTPUT-NEXT:succeeded(2)
    try_io(io_goal(1), R6, !IO),
    show(R6, !IO),
    % OUTPUT-NEXT:io_goal
    % OUTPUT-NEXT:int 9
    try_io(io_goal(9), R7, !IO),
    show(R7, !IO),
    % OUTPUT-NEXT:succeeded(-1)
    try(inner_catches, R8),
    show(R8, !IO),
    % OUTPUT-NEXT:my_error 4 big
    try(inner_rethrows, R9),
    show(R9, !IO),
    % OUTPUT-NEXT:succeeded(1)
    try(cc_multi_goal, R10),
    show(R10, !IO),
    % OUTPUT-NEXT:succeeded(2)
    try(cc_nondet_goal(2), R11),
    show(R11, !IO),
    % OUTPUT-NEXT:failed
    try(cc_nondet_goal(0), R12),
    show(R12, !IO),
    % OUTPUT-NEXT:int 6
    try(cc_nondet_goal(6), R13),
    show(R13, !IO),
    % OUTPUT-NEXT:cc_io_goal
    % OUTPUT-NEXT:succeeded(1)
    try_io(cc_io_goal, R14, !IO),
    show(R14, !IO),
    % OUTPUT-NEXT:2870
    io.write_int(list.foldl(int.plus, Kept, 0), !IO),
    io.nl(!IO),
    % OUTPUT-NEXT:before uncaught
    % OUTPUT-NEXT:Uncaught Merkurius exception:
    % OUTPUT-NEXT:my_error(3, "big")
    io.write_string("before uncaught\n", !IO),
    det_goal(3, W),
    io.write_int(W, !IO).
