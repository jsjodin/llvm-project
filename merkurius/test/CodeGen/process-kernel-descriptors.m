% The descriptors of a GPU process image (docs/INPROGRESS.md, GPU processes
% without omp target, step 4e): a descriptor that the device's compile copies
% from a library module brings those that its fields name, from the same
% module, as no library is linked into the image. down_info's fields are a
% monitor_ref and a down_reason, whose failed(string) names string; the
% host's code makes down_info's type_info, for a mailbox of them, and the
% image's module had down_info's descriptor without those it names, which
% LLVM's translation rejected. The program, which spawns at the host, runs
% without a GPU.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -O0 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t | FileCheck %s --match-full-lines --strict-whitespace
:- module process_kernel_descriptors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

    % Sends Reply the number it receives, plus one.
:- process adder(mailbox(int)) receives int on gpu.

adder(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        send(Reply, N + 1, !IO)
    else
        true
    ).

:- func id(down_info) = down_info.

id(D) = D.

main(!IO) :-
    new_mailbox(Me, !IO),
    new_mailbox(Downs, !IO),
    spawn(host, adder(Me), A, !IO),
    monitor(A, Downs, id, _, !IO),
    send(A, 41, !IO),
    receive(Me, any, infinity, R, !IO),
    ( if R = message(N) then
        io.write_int(N, !IO),
        io.nl(!IO)
    else
        true
    ),
    receive(Downs, any, infinity, D, !IO),
    ( if D = message(Info) then
        io.write_string(reason_text(reason(Info)), !IO),
        io.nl(!IO)
    else
        true
    ).
% CHECK:42
% CHECK-NEXT:normal
% DEVICE-DAG:@"merk.type_ctor.process.down_info/0" = linkonce_odr
% DEVICE-DAG:@"merk.type_ctor.process.monitor_ref/0" = linkonce_odr
% DEVICE-DAG:@"merk.type_ctor.process.down_reason/0" = linkonce_odr
% DEVICE-DAG:@"merk.type_ctor.string/0" = linkonce_odr
