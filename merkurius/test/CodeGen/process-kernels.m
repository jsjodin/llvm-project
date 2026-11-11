% Process kinds that run on a GPU (docs/INPROGRESS.md, GPU processes without
% omp target, step 4b, and GPU processes that run loops, one kernel for a
% GPU's processes). The device's compile, -fprocess-kernels, makes a start
% merk_process_<kind> for each such kind, which takes the address of the
% process's argument words, the kind's arguments and its mailbox's handle,
% and calls the kind's predicate; and merk_process_dispatch, which starts a
% kind by its number, its place among the module's GPU kinds. It keeps only
% what merk_process_dispatch reaches: not main, nor the kinds that run on
% the host, and hides it all. The host's compile embeds the image it is
% given and registers it at start-up, with the kinds' names in their order,
% and the message types that the device's code may send.
%
% The image exports its one kernel, merk_place, from its start object,
% libc's RPC client and the device's number, which the offload plugin and
% the runtime look up, and has no undefined dynamic symbol, with which HSA's
% loader would not load it. Without -O,
% the image is linked at -O2. At -O0 its link fails on an AMDGPU bug
% (process-kernels-O0.m). process-spawn-at.m runs the program.
%
% REQUIRES: amdgpu-registered-target
% RUN: merkc -fc1 -O0 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=KERNEL0 --implicit-check-not=main --implicit-check-not=doubler --match-full-lines --strict-whitespace
% RUN: merkc -fc1 -O2 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=KERNEL2 --implicit-check-not=main --implicit-check-not=doubler --match-full-lines --strict-whitespace
% RUN: printf 'IMAGE' > %t.elf
% RUN: merkc -fc1 -O0 -fprocess-declarations -fembed-process-image=gfx1030:%t.elf -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST --match-full-lines --strict-whitespace
% RUN: merkc -fc1 -O2 -fprocess-declarations -fembed-process-image=gfx1030:%t.elf -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST --match-full-lines --strict-whitespace
% RUN: rm -rf %t.dir && mkdir %t.dir
% RUN: merkc -c -fprocess-declarations --offload-arch=gfx1030 -save-temps=obj %s -o %t.dir/p.o
% RUN: llvm-readelf --dyn-syms %t.dir/process-kernels-gfx1030-processes.elf | FileCheck %s --check-prefix=IMAGE
% RUN: llvm-readelf --dyn-syms %t.dir/process-kernels-gfx1030-processes.elf | not grep -E ' UND +[^ ]' 

:- module process_kernels.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, process.

    % Adds up the numbers it receives until 0, then sends the sum, scaled,
    % to Reply.
:- process summer(int, float, mailbox(int)) receives int on gpu.

summer(Sum, Scale, Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N \= 0 then
        summer(Sum + N, Scale, Reply, Self, !IO)
    else
        send(Reply, truncate_to_int(float(Sum) * Scale), !IO)
    ).

    % A kind that runs on the host only.
:- process doubler(mailbox(int)) receives int.

doubler(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        send(Reply, 2 * N, !IO)
    else
        true
    ).

main(!IO) :-
    new_mailbox(Me, !IO),
    spawn_summer(1, 0.5, Me, S, !IO),
    send(S, 20, !IO),
    send(S, 63, !IO),
    send(S, 0, !IO),
    receive(Me, any, infinity, R1, !IO),
    ( if R1 = message(N1) then
        io.write_int(N1, !IO), io.nl(!IO),
        spawn_doubler(Me, D, !IO),
        send(D, N1, !IO),
        receive(Me, any, infinity, R2, !IO),
        ( if R2 = message(N2) then
            io.write_int(N2, !IO), io.nl(!IO)
        else
            true
        )
    else
        true
    ).

% The start takes the address of the process's words: the kind's integer and
% float, and each mailbox's handle as five words, which it puts in a cell on
% its stack.
% KERNEL0:define internal void @merk_process_summer(i64 %0) {
% KERNEL0-NEXT:  %2 = inttoptr i64 %0 to ptr
% KERNEL0-NEXT:  %3 = getelementptr i64, ptr %2, i32 0
% KERNEL0-NEXT:  %4 = load i64, ptr %3, align 8
% KERNEL0-NEXT:  %5 = getelementptr i64, ptr %2, i32 1
% KERNEL0-NEXT:  %6 = load i64, ptr %5, align 8
% KERNEL0-NEXT:  %7 = bitcast i64 %6 to double
% KERNEL0-NEXT:  %8 = alloca [5 x i64], i64 1, align 8, addrspace(5)
% KERNEL0-NEXT:  %9 = addrspacecast ptr addrspace(5) %8 to ptr
% KERNEL0:  %25 = alloca [5 x i64], i64 1, align 8, addrspace(5)
% KERNEL0:  %40 = getelementptr i64, ptr %2, i32 11
% KERNEL0:  call void @process_kernels.summer_6_0(i64 %4, double %7, ptr %9, ptr %26)
% KERNEL0-NEXT:  ret void
% KERNEL0-NEXT:}
% KERNEL0-EMPTY:
% KERNEL0-NEXT:define hidden void @merk_process_dispatch(i64 %0, i64 %1) {
% KERNEL0-NEXT:  switch i64 %0, label %4 [
% KERNEL0-NEXT:    i64 0, label %3
% KERNEL0-NEXT:  ]
% KERNEL0-EMPTY:
% KERNEL0-NEXT:3:                                                ; preds = %2
% KERNEL0-NEXT:  call void @merk_process_summer(i64 %1)
% KERNEL0-NEXT:  ret void
% KERNEL2:define hidden void @merk_process_dispatch(i64 %0, i64 %1) local_unnamed_addr {

% The host embeds the image, and registers it, after the message types.
% HOST:@process_kernels.merk_process_image.module = internal constant [16 x i8] c"process_kernels\00", align 1
% HOST-NEXT:@process_kernels.merk_process_image.kinds = internal constant [7 x i8] c"summer\00", align 1
% HOST-NEXT:@process_kernels.merk_process_image.gfx1030.gpu = internal constant [8 x i8] c"gfx1030\00", align 1
% HOST-NEXT:@process_kernels.merk_process_image.gfx1030 = internal constant [5 x i8] c"IMAGE", align 16
% HOST-NEXT:@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @process_kernels.merk_register_message_types.c, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @process_kernels.merk_register_process_images.c, ptr null }]
% HOST:  {{(tail )?}}call void @merk_register_message_type(i64 -7556727498917921977, ptr {{(nonnull )?}}@"merk.type_ctor.int/0")
% HOST:  {{(tail )?}}call void @merk_register_process_image(ptr {{(nonnull )?}}@process_kernels.merk_process_image.module, ptr {{(nonnull )?}}@process_kernels.merk_process_image.gfx1030.gpu, ptr {{(nonnull )?}}@process_kernels.merk_process_image.gfx1030, i64 5, ptr {{(nonnull )?}}@process_kernels.merk_process_image.kinds)

% The image exports the kernel and what the plugin and the runtime look up.
% IMAGE-DAG:{{.*}} GLOBAL PROTECTED {{.*}} merk_place
% IMAGE-DAG:{{.*}} WEAK   PROTECTED {{.*}} merk_device_number
% IMAGE-DAG:{{.*}} GLOBAL PROTECTED {{.*}} __llvm_rpc_client
% IMAGE-NOT:summer
