% A root that no liveness bundle lists, whose type comes from the call that
% made it (Lowering, Lowerer::notifyOperationReplaced; Statepoints,
% RecordTypes). bits/1 only reads its array's pointer, so at -O2 LLVM may
% sink its call after the allocation of the pair that holds its result,
% which keeps the array live across that allocation's safepoint, where
% liveness did not list it. That stopped the compile with "a root ... has
% no type at a safepoint". The loop collects many times while such arrays
% are live, and their elements must survive.
%
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: %t | FileCheck %s --match-full-lines --strict-whitespace

:- module gc_root_call_result.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string.

:- type answer
    --->    answer(int, int).

    % The low bits of the array's pointer, which are 0.
:- func bits(array(T)) = int.
:- pragma foreign_proc("C",
    bits(A::in) = (B::out),
    [will_not_call_mercury, promise_pure],
"
    B = (MR_Integer) (((unsigned long) A) & 7);
").

:- pred make(int::in, answer::out) is det.
make(N, R) :-
    array.init(1000, N, A),
    array.init(500, N + 1, C),
    R = answer(bits(A) + bits(C), A ^ elem(999) + C ^ elem(499)).

:- pred call_it(pred(int, answer)::in(pred(in, out) is det), int::in,
    answer::out) is det.
:- pragma no_inline(call_it/3).
call_it(P, N, R) :-
    P(N, R).

:- pred loop(int::in, int::in, int::in, int::out) is det.
loop(N, Last, !Sum) :-
    ( if N > Last then
        true
    else
        call_it(make, N, answer(Bits, Elements)),
        !:Sum = !.Sum + Bits + Elements,
        loop(N + 1, Last, !Sum)
    ).

main(!IO) :-
    loop(1, 20000, 0, Sum),
    io.format("%d\n", [i(Sum)], !IO).

% CHECK:400040000
