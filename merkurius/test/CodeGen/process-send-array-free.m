% A GPU process that sends arrays as whole messages frees each after its
% send, which copies it (Frees.cpp): a sender of eight arrays of 2^17
% floats, each in a block of 2 MiB from the image's cache of large blocks
% (runtime/device/device.cpp, RawAllocate), holds one of them at a time,
% where it held all eight before. The kind runs at the host here, and on
% the GPU in process-send-array-free-gpu.test, which checks the most the
% process held.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_send_array_free.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

:- process sender(mailbox(array(float)), int) receives int on gpu.

sender(To, N, Self, !IO) :-
    ( if N > 0 then
        A = array.init(131072, float(N)),
        send(To, A, !IO),
        sender(To, N - 1, Self, !IO)
    else
        true
    ).

:- pred take(mailbox(array(float))::in, int::in, float::in, float::out,
    io::di, io::uo) is det.

take(From, N, !Sum, !IO) :-
    ( if N > 0 then
        receive(From, any, after(10000), R, !IO),
        ( if R = message(A) then
            !:Sum = !.Sum + array.lookup(A, 0) +
                float(array.size(A))
        else
            true
        ),
        take(From, N - 1, !Sum, !IO)
    else
        true
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(M, !IO),
    spawn(Place, sender(M, 8), _, !IO),
    take(M, 8, 0.0, Sum, !IO),
    io.format("%.1f\n", [f(Sum)], !IO).
% CHECK:1048612.0
