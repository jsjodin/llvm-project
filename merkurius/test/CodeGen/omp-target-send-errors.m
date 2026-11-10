% A target region sends a message of any ground type whose arguments are
% not higher-order (docs/INPROGRESS.md, GPU processes without omp target,
% step 4e), as omp-target-term-messages.m does. A message of a type with a
% closure in its arguments, which cannot cross between the host and a GPU,
% is an error at the send, in the host's compile as in the device's, since
% both compile the region. A closure in a constructor's field is found when
% the runtime copies the message (process-gpu-messages.m).
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
% CHECK: omp-target-send-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for sending messages of type list.list(pred(int)) in target regions

main(!IO) :-
    new_mailbox(Tests, !IO),
    tests(Tests, !IO).
