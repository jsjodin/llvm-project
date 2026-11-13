% Tail calls between procedures of different types on an AMD GPU, which has
% no tail calling convention, so that LLVM takes a musttail call only
% between functions of one type (design document, "Calling convention";
% Lowering.cpp, UnifyTailCallTypes). A GPU process's states, even and odd,
% take different arguments, a float and a character beside the counts, and
% call each other last a million times, which a stack of ordinary calls
% would not hold: the procedures that such calls join get one type, each
% type's parameters as many as the procedure with the most, the rest poison
% at their calls, and the calls are musttail. A procedure the interface
% exports keeps its type, which other modules call it by, so its call stays
% an ordinary one, which the recursion warning reports. So does the call of
% the kind from the process's entry, which passes it the mailbox it builds
% in its own frame, which a tail call would end. The program runs its kind
% at the host here, and at the GPU in gpu-tail-calls-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -O0 -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
%
% Only the exported procedure's pair of calls is reported.
% DEVICE: gpu-tail-calls.m:98:9: warning: this call may recurse, and it is not a tail call
% DEVICE-NEXT: gpu-tail-calls.m:105:5: warning: this call may recurse
% DEVICE-NOT: warning:
%
% The exported procedure keeps its type, and so its partner its own.
% DEVICE: define hidden void @gpu_tail_calls.count_down_{{[0-9_]+}}(ptr %0, i64 %1, i64 %2)
% DEVICE: call void @gpu_tail_calls.step_down_{{[0-9_]+}}(ptr %0, double 1.000000e+00, i64 %1, i64 %2)
% DEVICE: define internal void @gpu_tail_calls.step_down_{{[0-9_]+}}(ptr %0, double %1, i64 %2, i64 %3)
%
% even and odd have one type, the mailbox, the two counts, the float and the
% character, and call each other with musttail, even's calls passing poison
% for the two it does not have.
% DEVICE: call void @gpu_tail_calls.even_{{[0-9_]+}}(ptr %0, i64 0, i64 1000000, double poison, i32 poison)
% DEVICE-LABEL: define internal void @gpu_tail_calls.even_{{[0-9_]+}}(ptr %0, i64 %1, i64 %2, double %3, i32 %4)
% DEVICE: musttail call void @gpu_tail_calls.odd_{{[0-9_]+}}(ptr %0, i64 %1, i64 %{{[0-9]+}}, double 5.000000e-01, i32 98)
% DEVICE-LABEL: define internal void @gpu_tail_calls.odd_{{[0-9_]+}}(ptr %0, i64 %1, i64 %2, double %3, i32 %4)
% DEVICE: musttail call void @gpu_tail_calls.even_{{[0-9_]+}}(ptr %0, i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, double poison, i32 poison)
%
% DEVICE-LABEL: define internal void @merk_process_states(i64 %0)
% DEVICE-NOT: musttail
% DEVICE: call void @gpu_tail_calls.states_{{[0-9_]+}}(ptr %{{[0-9]+}}, ptr %{{[0-9]+}})
% DEVICE-NEXT: ret void

% CHECK:1000000
% CHECK-NEXT:3

:- module gpu_tail_calls.
:- interface.
:- import_module io, process.
:- pred main(io::di, io::uo) is det.

    % Counts N down, adding 1 to Sum each time, by a tail call of a local
    % procedure.
:- pred count_down(mailbox(int)::in, int::in, int::in, io::di, io::uo)
    is det.

:- implementation.
:- import_module char, float, int, list.

:- process states(mailbox(int)) receives int on gpu.

states(Reply, _Self, !IO) :-
    even(Reply, 0, 1000000, !IO),
    count_down(Reply, 0, 3, !IO).

    % Adds nothing on even steps, and on odd ones the float and the
    % character's code less 'a''s, 0.5 and 1 here.
:- pred even(mailbox(int)::in, int::in, int::in, io::di, io::uo) is det.

even(Reply, Sum, N, !IO) :-
    ( if N = 0 then
        send(Reply, Sum, !IO)
    else
        odd(Reply, 0.5, Sum, N - 1, 'b', !IO)
    ).

:- pred odd(mailbox(int)::in, float::in, int::in, int::in, char::in,
    io::di, io::uo) is det.

odd(Reply, Half, Sum, N, C, !IO) :-
    ( if N = 0 then
        send(Reply, Sum, !IO)
    else
        even(Reply, Sum + truncate_to_int(Half * 2.0) +
            char.to_int(C) - char.to_int('a'), N - 1, !IO)
    ).

count_down(Reply, Sum, N, !IO) :-
    ( if N = 0 then
        send(Reply, Sum, !IO)
    else
        step_down(Reply, 1.0, Sum, N, !IO)
    ).

:- pred step_down(mailbox(int)::in, float::in, int::in, int::in,
    io::di, io::uo) is det.

step_down(Reply, One, Sum, N, !IO) :-
    count_down(Reply, Sum + truncate_to_int(One), N - 1, !IO).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Reply, !IO),
    spawn(Place, states(Reply), _, !IO),
    receive(Reply, any, after(60000), R1, !IO),
    receive(Reply, any, after(60000), R2, !IO),
    ( if R1 = message(A), R2 = message(B) then
        io.write_int(A, !IO),
        io.nl(!IO),
        io.write_int(B, !IO),
        io.nl(!IO)
    else
        io.write_string("timeout\n", !IO)
    ).
