% Process kinds that run on a GPU (docs/INPROGRESS.md, GPU processes without
% omp target, step 4b). The kernels' compile, -fprocess-kernels, makes a
% kernel merk_process_<kind> for each such kind, which takes the kind's
% arguments and its mailbox's handle by value and calls the kind's predicate,
% and keeps only what the kernels reach: not main, nor the kinds that run on
% the host. It hides the rest, so that the image exports only the kernels.
% The host's compile embeds the image it is given and registers it at
% start-up, with the message types that the device's code may send.
%
% The program runs its kinds on the host: spawning on a GPU is step 4c. Its
% image is linked all the same, and exports the kernel, libc's RPC client
% and the device's number, which the offload plugin and the runtime look up.
% Without -O, the image is linked at -O2. At -O0 its link fails on an
% AMDGPU bug (process-kernels-O0.m).
%
% REQUIRES: amdgpu-registered-target
% RUN: merkc -fc1 -O0 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=KERNEL0 --implicit-check-not=main --implicit-check-not=doubler --match-full-lines --strict-whitespace
% RUN: merkc -fc1 -O2 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=KERNEL2 --implicit-check-not=main --implicit-check-not=doubler --match-full-lines --strict-whitespace
% RUN: printf 'IMAGE' > %t.elf
% RUN: merkc -fc1 -O0 -fprocess-declarations -fembed-process-image=gfx1030:%t.elf -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST --match-full-lines --strict-whitespace
% RUN: merkc -fc1 -O2 -fprocess-declarations -fembed-process-image=gfx1030:%t.elf -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST --match-full-lines --strict-whitespace
% RUN: rm -rf %t.dir && mkdir %t.dir
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 -save-temps=obj %s -o %t.dir/p2 && %t.dir/p2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: llvm-readelf --dyn-syms %t.dir/process-kernels-gfx1030-processes.elf | FileCheck %s --check-prefix=IMAGE

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
% CHECK:42
% CHECK-NEXT:84

% The kernel takes the kind's integer and float by value, and each mailbox's
% handle as five words, which it puts in a cell on its stack.
% KERNEL0:define protected amdgpu_kernel void @merk_process_summer(i64 %0, double %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11) {
% KERNEL0-NEXT:  %13 = alloca [5 x i64], i64 1, align 8, addrspace(5)
% KERNEL0-NEXT:  %14 = addrspacecast ptr addrspace(5) %13 to ptr
% KERNEL0-NEXT:  %15 = getelementptr i64, ptr %14, i32 0
% KERNEL0-NEXT:  store i64 %2, ptr %15, align 8
% KERNEL0:  store i64 %6, ptr %19, align 8
% KERNEL0-NEXT:  %20 = alloca [5 x i64], i64 1, align 8, addrspace(5)
% KERNEL0-NEXT:  %21 = addrspacecast ptr addrspace(5) %20 to ptr
% KERNEL0:  store i64 %11, ptr %26, align 8
% KERNEL0-NEXT:  call void @process_kernels.summer_6_0(i64 %0, double %1, ptr %14, ptr %21)
% KERNEL0-NEXT:  ret void
% KERNEL0-NEXT:}
% KERNEL2:define protected amdgpu_kernel void @merk_process_summer(i64 %0, double %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11) local_unnamed_addr {

% The host embeds the image, and registers it, after the message types.
% HOST:@process_kernels.merk_process_image.module = internal constant [16 x i8] c"process_kernels\00", align 1
% HOST-NEXT:@process_kernels.merk_process_image.gfx1030.gpu = internal constant [8 x i8] c"gfx1030\00", align 1
% HOST-NEXT:@process_kernels.merk_process_image.gfx1030 = internal constant [5 x i8] c"IMAGE", align 16
% HOST-NEXT:@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @process_kernels.merk_register_message_types.c, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @process_kernels.merk_register_process_images.c, ptr null }]
% HOST:  {{(tail )?}}call void @merk_register_message_type(i64 -7556727498917921977, ptr {{(nonnull )?}}@"merk.type_ctor.int/0")
% HOST:  {{(tail )?}}call void @merk_register_process_image(ptr {{(nonnull )?}}@process_kernels.merk_process_image.module, ptr {{(nonnull )?}}@process_kernels.merk_process_image.gfx1030.gpu, ptr {{(nonnull )?}}@process_kernels.merk_process_image.gfx1030, i64 5)

% The image exports the kernel and what the plugin and the runtime look up.
% IMAGE-DAG:{{.*}} GLOBAL PROTECTED {{.*}} merk_process_summer
% IMAGE-DAG:{{.*}} WEAK   PROTECTED {{.*}} merk_device_number
% IMAGE-DAG:{{.*}} GLOBAL PROTECTED {{.*}} __llvm_rpc_client
% IMAGE-NOT:summer_6_0
