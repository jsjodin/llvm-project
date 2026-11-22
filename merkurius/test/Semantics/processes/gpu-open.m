% Open terms are host-only (design document, "Open terms as built"):
% code that runs on a GPU, a process kind's that runs on a GPU, what it
% calls, and a target region, may not use a value whose type has open(T) or
% closed(T) in it. A kind that runs on the host may. A GPU kind declared to
% receive such messages is reported at its declaration, even if it never
% receives, and its receives do not repeat it.
%
% RUN: not merkc -fc1 -fopen-types -fprocess-declarations -fopenmp -fdebug-dump-determinism %s 2>&1 >/dev/null \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module gpu_open.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, open, process.

:- process device_kind receives int on gpu.
device_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO),
    make(_).
% CHECK: gpu-open.m:[[@LINE-1]]:5: error: open terms are host-only, so code that runs on a GPU may not use a value of type open.open(list.list(int))

    % What a GPU process calls is checked where it is defined.
:- pred make(open(list(int))::oa) is det.
make([_ | _]).
% CHECK: gpu-open.m:[[@LINE-1]]:6: error: open terms are host-only, so code that runs on a GPU may not use a value of type open.open(list.list(int))

:- process open_kind receives open(int) on gpu.
% CHECK: gpu-open.m:[[@LINE-1]]:4: error: open terms are host-only, so a process kind that runs on a GPU may not receive messages of type open.open(int)
open_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

:- process closed_kind receives closed(int) on gpu.
% CHECK: gpu-open.m:[[@LINE-1]]:4: error: open terms are host-only, so a process kind that runs on a GPU may not receive messages of type open.closed(int)
closed_kind(_, !IO).

:- process host_kind receives int.
host_kind(Self, !IO) :-
    receive(Self, any, infinity, _, !IO),
    make(_).

main(!IO) :-
    %$omp target
    (
        C = closed(1),
        io.write(C, !IO)
    ).
% CHECK: gpu-open.m:[[@LINE-3]]:13: error: open terms are host-only, so code that runs on a GPU may not use a value of type open.closed(int)
% CHECK: gpu-open.m:[[@LINE-3]]:9: error: open terms are host-only, so code that runs on a GPU may not use a value of type open.closed(int)
