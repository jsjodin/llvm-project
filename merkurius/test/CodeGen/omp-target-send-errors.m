% A target region sends and receives only flat messages (design document,
% "Messages and memory on the device"; docs/INPROGRESS.md, GPU processes
% without omp target, step 5a): a message of a type that is recursive, or
% holds a closure, is an error at the send or the receive, in the host's
% compile as in the device's, since both compile the region. The region's
% own code may make such values, which go when it ends. The other cases are
% in test/Semantics/processes/gpu-flat.m.
%
% RUN: not merkc -fc1 -fopenmp -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:
% REQUIRES: amdgpu-registered-target

:- module omp_target_send_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, process.

:- pred tests(mailbox(list(pred(int)))::in, io::di, io::uo) is det.
tests(Tests, !IO) :-
    %$omp target
    send(Tests, [], !IO).
% CHECK: omp-target-send-errors.m:[[@LINE-1]]:5: error: a message that code running on a GPU sends or receives must be flat, but this one is of type list.list(pred(int)): pred(int) is a higher-order type

:- pred lists(mailbox(list(int))::in, io::di, io::uo) is det.
lists(Lists, !IO) :-
    %$omp target
    (
        receive(Lists, any, after(0), R, !IO),
        % CHECK: omp-target-send-errors.m:[[@LINE-1]]:9: error: a message that code running on a GPU sends or receives must be flat, but this one is of type list.list(int): list.list(int) is recursive
        L = ( if R = message(M) then M else [1, 2] ),
        send(Lists, L, !IO)
        % CHECK: omp-target-send-errors.m:[[@LINE-1]]:9: error: a message that code running on a GPU sends or receives must be flat, but this one is of type list.list(int): list.list(int) is recursive
    ).

main(!IO) :-
    new_mailbox(Tests, !IO),
    tests(Tests, !IO),
    new_mailbox(Lists, !IO),
    lists(Lists, !IO).
