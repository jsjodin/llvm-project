% The stack's overflow check (design document, "Processes and scheduler"): a
% procedure with a call that is not a tail call in a cycle of calls has it,
% including a cycle that goes through a tail call, and so does one that calls
% another module. So does a tail recursion, which a preempted process must
% stop in. A leaf does not. Running out of stack ends the program with an
% error, after what it wrote.
%
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% The checks come after the optimizer, and a check may yield to other
% processes, so what it inferred of the checked procedures and their callers,
% that they touch no memory, do not synchronize and return, is dropped, but
% not from intrinsics, which are nocallback.
% RUN: merkc -fc1 -emit-llvm -O2 -triple x86_64-unknown-linux-gnu %s -o %t.ll
% RUN: grep "^attributes" %t.ll | grep -v nocallback | FileCheck %s --check-prefix=INFERRED
% INFERRED: attributes
% INFERRED-NOT: memory(
% INFERRED-NOT: nosync
% INFERRED-NOT: willreturn
% RUN: merkc -O0 %s -o %t.0 && not %t.0 2> %t.0.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR --match-full-lines --strict-whitespace < %t.0.err
% RUN: merkc -O2 %s -o %t.2 && not %t.2 2> %t.2.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR --match-full-lines --strict-whitespace < %t.2.err

:- module stack_check.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

% LLVM-LABEL: define tailcc void @stack_check.main_2_0()
% LLVM: call void @merk_stack_limit_passed()
main(!IO) :-
    io.write_int(deep(1000), !IO), io.nl(!IO),
    io.write_int(through_tail(1000), !IO), io.nl(!IO),
    io.write_int(count(1000000, 0), !IO), io.nl(!IO),
    io.write_int(deep(100000000), !IO), io.nl(!IO),
    io.write_int(through_tail(100000000), !IO), io.nl(!IO).

% LLVM-LABEL: define internal tailcc i64 @stack_check.deep_f1_0(
% LLVM: call void @merk_stack_limit_passed()
:- func deep(int) = int.
deep(N) = (if N = 0 then 0 else (deep(N - 1) * 3 + N) mod 7).

% LLVM-LABEL: define internal tailcc i64 @stack_check.through_tail_f1_0(
% LLVM: call void @merk_stack_limit_passed()
:- func through_tail(int) = int.
through_tail(N) = (if N = 0 then 0 else (tail_back(N - 1) * 3 + N) mod 7).

% LLVM-LABEL: define internal tailcc i64 @stack_check.count_f2_0(
% LLVM: call void @merk_stack_limit_passed()
:- func count(int, int) = int.
count(N, A) = (if N = 0 then A else count(N - 1, leaf(A))).

% LLVM-LABEL: define internal tailcc i64 @stack_check.tail_back_f1_0(
% LLVM: call void @merk_stack_limit_passed()
:- func tail_back(int) = int.
tail_back(N) = through_tail(N).

% LLVM-LABEL: define internal tailcc i64 @stack_check.leaf_f1_0(
% LLVM-NOT: merk_stack_limit_passed()
% LLVM: define void @merk_main()
:- func leaf(int) = int.
leaf(N) = N + 1.

% OUTPUT:1
% OUTPUT-NEXT:1
% OUTPUT-NEXT:1000000
% OUTPUT-NOT:{{.}}

% ERROR:Mercury runtime: stack overflow
