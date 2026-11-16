% A procedure that makes, fills or resizes an array of flat records passes
% the record's bytes to the runtime from a stack slot, zeroed first for its
% padding (Lowering, RecordSlot). The memset that zeroes it was taken for an
% escape of the slot's address, so every tail call of the procedure became
% an ordinary call: a loop that made such arrays grew the stack by a frame
% each time round, on the host as in a GPU process, whose stack is small. A
% million iterations here would overflow the host's stack. The output is
% mmc's.
%
% RUN: merkc -fc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -O2 %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.O0 && %t.O0 | FileCheck %s --match-full-lines --strict-whitespace

:- module record_array_tail_calls.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

:- type pt ---> pt(x :: float, n :: int).

:- pred loop(int::in, int::in, int::out) is det.
loop(I, !Sum) :-
    ( if I > 0 then
        array.init(I mod 3 + 1, pt(float(I), I), A0),
        array.fill(pt(0.5, I mod 7), A0, A1),
        array.resize(I mod 5 + 1, pt(1.5, 2), A1, A),
        array.lookup(A, array.size(A) - 1) = pt(_, N),
        !:Sum = !.Sum + N,
        loop(I - 1, !Sum)
    else
        true
    ).
% LLVM-LABEL: define internal tailcc i64 @record_array_tail_calls.loop_3_0(
% LLVM: musttail call tailcc i64 @record_array_tail_calls.loop_3_0(

main(!IO) :-
    loop(1000000, 0, Sum),
    io.write_int(Sum, !IO),
    io.nl(!IO).

% CHECK:2400006
% CHECK-EMPTY:
