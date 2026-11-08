% The address of the thread's heap is taken again after a call, across which
% the process may have moved to another thread (docs/INPROGRESS.md, The
% scheduler, D1), also when the optimizer has run. It is an asm with a side
% effect, as the backend moves the thread pointer out of loops and across calls
% otherwise.
%
% RUN: merkc -fc1 -O2 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s

:- module heap_address.
:- interface.
:- import_module io, list.
:- pred twice(int::in, list(int)::out, io::di, io::uo) is det.
:- implementation.
:- import_module int.

% CHECK-LABEL: @heap_address.twice_4_0(
% CHECK: call ptr asm sideeffect "movq %fs:0, $0\0A\09leaq merk_heap@tpoff($0), $0", "=r"()
% CHECK: call tailcc {{.*}}@io.write_line
% CHECK: call ptr asm sideeffect "movq %fs:0, $0\0A\09leaq merk_heap@tpoff($0), $0", "=r"()
% CHECK-NOT: merk.heap_address
twice(N, L2, !IO) :-
    L1 = [N],
    io.write_line(L1, !IO),
    L2 = [N + 1 | L1].
