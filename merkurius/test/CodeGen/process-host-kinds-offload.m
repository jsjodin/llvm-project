% A module whose process kinds all run on the host, compiled for a GPU
% (docs/INPROGRESS.md, GPU processes without omp target, step 4b): its image
% of GPU processes has no kinds, which the host's compile leaves out, but
% the image's kernel still calls merk_process_dispatch, which the device's
% compile makes for it. The image's link failed with that symbol undefined.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: rm -rf %t.dir && mkdir %t.dir
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 -save-temps=obj %libomp-flags %s -o %t.dir/p
% RUN: %libomp-env %t.dir/p | FileCheck %s --match-full-lines --strict-whitespace
% RUN: llvm-readelf --dyn-syms %t.dir/process-host-kinds-offload-gfx1030-processes.elf | not grep -E ' UND +[^ ]'
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module process_host_kinds_offload.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

    % Answers a number with one more.
:- process adder(mailbox(int)) receives int.

adder(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then send(Reply, N + 1, !IO) else true ).

main(!IO) :-
    new_mailbox(Me, !IO),
    spawn(adder(Me), A, !IO),
    send(A, 41, !IO),
    receive(Me, any, after(10000), R, !IO),
    ( if R = message(N) then io.write_int(N, !IO) else io.print(R, !IO) ),
    io.nl(!IO).
% CHECK:42
