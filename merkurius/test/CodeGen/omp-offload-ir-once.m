% The host's compile with offloading writes its LLVM IR as bitcode, which
% the device's compile reads, and reads it back to embed the device's code
% (-x ir). The optimizer, the stack checks and the heap's addresses run on
% it once: the IR read back is compiled as it is. A second run of the
% optimizer folded the inttoptr that the first makes at its end of a number
% that ask's closure has as a word, and the number, carried on as a heap
% pointer, was a root without a type at a safepoint, which crashed the
% compile at -O2.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -emit-llvm-bc -O2 -fopenmp %s -o %t.bc
% RUN: merkc -fc1 -emit-llvm -O2 -fopenmp -x ir %t.bc -o %t.again.ll
% RUN: llvm-dis %t.bc -o %t.first.ll
% RUN: grep -v -e '^; ModuleID' -e '^source_filename' %t.first.ll > %t.first
% RUN: grep -v -e '^; ModuleID' -e '^source_filename' %t.again.ll > %t.again
% RUN: diff %t.first %t.again
% RUN: merkc -O2 -fopenmp --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fopenmp --offload-arch=gfx1030 %libomp-flags %s -o %t0
% RUN: %libomp-env %t0 | FileCheck %s --match-full-lines --strict-whitespace

:- module omp_offload_ir_once.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process.

main(!IO) :-
    new_mailbox(Me, !IO),
    list.foldl(ask(Me), [1, 2], !IO).

    % Sends N + 1 and N to itself, and prints N and what it receives.
:- pred ask(mailbox(int)::in, int::in, io::di, io::uo) is det.

ask(Me, N, !IO) :-
    send(Me, N + 1, !IO),
    send(Me, N, !IO),
    receive(Me, any, after(60000), R1, !IO),
    receive(Me, any, after(60000), R2, !IO),
    io.write_int(N, !IO),
    list.foldl(print_answer, [R1, R2], !IO),
    io.nl(!IO).

:- pred print_answer(received(int)::in, io::di, io::uo) is det.

print_answer(R, !IO) :-
    io.write_string(" ", !IO),
    ( if R = message(V) then io.write_int(V, !IO) else io.print(R, !IO) ).
% CHECK:1 2 1
% CHECK-NEXT:2 3 2
