% Every team of a GPU processes' kernel runs processes (docs/INPROGRESS.md,
% GPU processes without omp target, step 4b): seventeen adders, all spawned
% before any is sent a number, fill more slots than a team has, and each
% answers. The kernel's teams must all be resident at once, as a slot waits
% for a process only once its team runs. The device runtime's Fail trapped,
% which made the kernel's stack dynamic, and the GPU then held back all its
% teams but the first, as it did not hold their scratch: the adders in the
% others never answered. Now the host also waits for the teams to start
% when it launches the kernel (process-teams-gpu.test).
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target
% RUN: rm -rf %t.dir && mkdir %t.dir
% RUN: merkc -c -fprocess-declarations --offload-arch=gfx1030 -save-temps=obj %s -o %t.dir/p.o
% RUN: llvm-readelf --notes %t.dir/process-teams-gfx1030-processes.elf | FileCheck %s --check-prefix=STACK
% STACK: .uses_dynamic_stack: false
% RUN: merkc -c -fprocess-declarations --offload-arch=gfx1030 -save-temps=obj %S/Inputs/process_teams_recursive.m -o %t.dir/r.o
% RUN: llvm-readelf --notes %t.dir/process_teams_recursive-gfx1030-processes.elf | FileCheck %s --check-prefix=DYNAMIC
% DYNAMIC: .uses_dynamic_stack: true
%
% The recursive module's calls that recurse other than by tail calls are
% warned of, and each procedure that makes one checks at its entry that the
% stack has room for what its calls take below its frame, which the backend
% knows, in the stack that the runtime sets.
% RUN: merkc -fc1 -O2 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %S/Inputs/process_teams_recursive.m -o %t.ll 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
% WARN: process_teams_recursive.m:20:33: warning: this call may recurse, and it is not a tail call, so each call takes more of the stack of a GPU process, which is limited (MERK_GPU_PROCESS_STACK_BYTES): a recursion deeper than it has room for ends the program
% WARN: process_teams_recursive.m:20:46: warning: this call may recurse
% WARN: process_teams_recursive.m:32:35: warning: this call may recurse
% RUN: FileCheck %s --check-prefix=CHECKED < %t.ll
% CHECKED: @merk_device_stack_bytes = external addrspace(1) global i32
% CHECKED-LABEL: define internal fastcc {{.*}}@process_teams_recursive.fib_f1_0(
% CHECKED-NEXT: %[[BYTES:[0-9]+]] = load i32, ptr addrspace(1) @merk_device_stack_bytes
% CHECKED-NEXT: %[[SP:[0-9]+]] = call ptr addrspace(5) @llvm.stacksave.p5()
% CHECKED-NEXT: %[[AT:[0-9]+]] = ptrtoint ptr addrspace(5) %[[SP]] to i32
% CHECKED-NEXT: %[[CALLEES:[0-9]+]] = call i32 @llvm.amdgcn.callee.stack.size()
% CHECKED-NEXT: %[[END:[0-9]+]] = add nuw i32 %[[AT]], %[[CALLEES]]
% CHECKED-NEXT: icmp ugt i32 %[[END]], %[[BYTES]]
% CHECKED: call void @merk_device_stack_overflow()
% CHECKED-LABEL: define internal fastcc {{.*}}@process_teams_recursive.deep_f1_0(
% CHECKED-NEXT: load i32, ptr addrspace(1) @merk_device_stack_bytes
% CHECKED: call void @merk_device_stack_overflow()
%
% What the optimizer inferred before the checks, that the procedures, and
% those that call them, touch no memory and need no flat scratch, no longer
% holds: the report of an overflow writes its stack through flat addresses,
% which faulted at 768 bytes a lane. The dispatch, which calls them, has no
% inferred attributes left, only its own, convergent and noinline, as every
% function of the compile is convergent and the kinds' code stays out of
% line; merk_process_loop, which runs no loop here, keeps its own.
% RUN: FileCheck %s --check-prefix=INFERRED < %t.ll
% INFERRED: define internal fastcc {{.*}}@process_teams_recursive.fib_f1_0({{.*}}) unnamed_addr #[[FIB:[0-9]+]] {
% INFERRED: define internal fastcc {{.*}}@process_teams_recursive.deep_f1_0({{.*}}) unnamed_addr #[[DEEP:[0-9]+]] {
% INFERRED: define hidden void @merk_process_dispatch(i64 %0, i64 %1) local_unnamed_addr #[[DISPATCH:[0-9]+]] {
% INFERRED-NOT: attributes #[[FIB]] = {{.*}}memory(none)
% INFERRED-NOT: attributes #[[FIB]] = {{.*}}amdgpu-no-
% INFERRED-NOT: attributes #[[DEEP]] = {{.*}}memory(none)
% INFERRED-NOT: attributes #[[DEEP]] = {{.*}}amdgpu-no-
% INFERRED: attributes #[[DISPATCH]] = { convergent noinline }
%
% The adders recurse only by tail calls: no warning, no check.
% RUN: merkc -fc1 -O2 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o %t.adders.ll 2>&1 | count 0
% RUN: not grep merk_device_stack %t.adders.ll
%
% The host registers the image with what its kernel's calls take outside
% the recursion, which the runtime checks against the stack it launches the
% kernel with.
% RUN: merkc -fc1 -fprocess-declarations -fembed-process-image=gfx1030:%t.dir/process_teams_recursive-gfx1030-processes.elf -emit-llvm %S/Inputs/process_teams_recursive.m -o - 2>/dev/null | FileCheck %s --check-prefix=FRAMES
% FRAMES: call void @merk_register_process_image({{.*}}, ptr @process_teams_recursive.merk_process_image.kinds, i64 {{[1-9][0-9]*}}, ptr @process_teams_recursive.merk_process_image.loops)

:- module process_teams.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

    % Sends Reply the number it receives, plus one, and ends.
:- process adder(mailbox(int)) receives int on gpu.

adder(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then send(Reply, N + 1, !IO) else true ).

main(!IO) :-
    new_mailbox(Me, !IO),
    Numbers = 1 .. 17,
    list.map_foldl(spawn_adder(Me), Numbers, Adders, !IO),
    send_numbers(Adders, Numbers, !IO),
    sum_answers(Me, list.length(Numbers), 0, Count, 0, Sum, !IO),
    io.format("%d answers, %d in all\n", [i(Count), i(Sum)], !IO).
% CHECK:17 answers, 170 in all

:- pred spawn_adder(mailbox(int)::in, int::in, mailbox(int)::out,
    io::di, io::uo) is det.

spawn_adder(Me, _, A, !IO) :-
    spawn(gpu(0), adder(Me), A, !IO).

:- pred send_numbers(list(mailbox(int))::in, list(int)::in,
    io::di, io::uo) is det.

send_numbers(As, Ns, !IO) :-
    ( if As = [A | As1], Ns = [N | Ns1] then
        send(A, N, !IO),
        send_numbers(As1, Ns1, !IO)
    else
        true
    ).

    % Receives Left more answers, each within ten seconds.
:- pred sum_answers(mailbox(int)::in, int::in, int::in, int::out, int::in,
    int::out, io::di, io::uo) is det.

sum_answers(Me, Left, !Count, !Sum, !IO) :-
    ( if Left = 0 then
        true
    else
        receive(Me, any, after(10000), R, !IO),
        ( if R = message(M) then
            !:Count = !.Count + 1,
            !:Sum = !.Sum + M,
            sum_answers(Me, Left - 1, !Count, !Sum, !IO)
        else
            true
        )
    ).
