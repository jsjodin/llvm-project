% Receive goals in a process kind that runs on a GPU (docs/INPROGRESS.md,
% Concurrency in merk, step 1): each is a merk.receive, which lowering makes
% a loop that calls the cases' matchers directly, and asks the device's
% runtime for the messages to try (Lowering.cpp, LowerSelect; device.cpp,
% merk_device_select_begin). A mailbox's messages are tried in the order
% they came, those that earlier receives skipped first, and each with every
% case of its mailbox in turn. A case may use a value of the process's, as
% Tag, and a receive may time out. The program runs its kind at the host
% here, through the library's receive_goal, and at the GPU in
% process-gpu-select-gpu.test, with the same output. The device's code
% makes no closure and calls nothing through a pointer, which would make
% the kernel assume the most registers.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -freceive-goal -fprocess-declarations -Wno-unselected-message --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -freceive-goal -fprocess-declarations -Wno-unselected-message --offload-arch=%offload-archs %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fc1 -emit-llvm -freceive-goal -fprocess-declarations -Wno-unselected-message -triple amdgpu10.30-amd-amdhsa -fprocess-kernels %s -o %t.ll
% RUN: FileCheck %s --check-prefix=LLVM < %t.ll
% RUN: FileCheck %s --check-prefix=POINTER < %t.ll

:- module process_gpu_select.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- type job
    --->    low(int)
    ;       high(int)
    ;       tagged(int, int)
    ;       stop.

:- type report
    --->    got_high(int)
    ;       got_low(int)
    ;       got_tagged(int)
    ;       timed_out
    ;       stopped.

    % Selects from its own mailbox, and reports to Report.
:- process selector(int, mailbox(report)) receives job on gpu.

selector(Tag, Report, Self, !IO) :-
    % high first, though messages came before it: they stay queued.
    receive [io(!IO)] (
        select(Self, high(H)) -> send(Report, got_high(H), !IO)
    ),
    % The lows the first receive skipped, in the order they came.
    receive [io(!IO)] (
        select(Self, low(L1)) -> send(Report, got_low(L1), !IO)
    ),
    receive [io(!IO)] (
        select(Self, low(L2)) -> send(Report, got_low(L2), !IO)
    ),
    % A pattern with the process's argument, Tag, in it.
    receive [io(!IO)] (
        select(Self, tagged(Tag, T)) -> send(Report, got_tagged(T), !IO)
    ),
    % Nothing is high now: the timeout arm runs.
    receive [timeout(after(20)), io(!IO)] (
        select(Self, high(H2)) -> send(Report, got_high(H2), !IO)
    ;
        timeout -> send(Report, timed_out, !IO)
    ),
    loop(Report, Self, !IO).

    % Three cases of one mailbox: each message is tried with each case
    % before the next message is, so tagged(8, 80), which came before
    % low(6), is taken first, then the low that came later, then stop.
:- pred loop(mailbox(report)::in, mailbox(job)::in, io::di, io::uo) is det.

loop(Report, Self, !IO) :-
    receive [io(!IO)] (
        select(Self, low(L)) ->
            send(Report, got_low(L), !IO),
            loop(Report, Self, !IO)
    ;
        select(Self, tagged(_, N)) ->
            send(Report, got_tagged(N), !IO),
            loop(Report, Self, !IO)
    ;
        select(Self, stop) ->
            send(Report, stopped, !IO)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Report, !IO),
    spawn(Place, selector(7, Report), Self, !IO),
    send(Self, low(1), !IO),
    send(Self, tagged(8, 80), !IO),
    send(Self, low(3), !IO),
    send(Self, tagged(7, 70), !IO),
    % Saved after tagged(8, 80): loop must take that one first.
    send(Self, low(6), !IO),
    send(Self, high(2), !IO),
    reports(5, Report, !IO),
    send(Self, low(5), !IO),
    send(Self, stop, !IO),
    reports(4, Report, !IO).

:- pred reports(int::in, mailbox(report)::in, io::di, io::uo) is det.

reports(N, Report, !IO) :-
    ( if N > 0 then
        receive(Report, any, after(5000), R, !IO),
        ( if R = message(M) then
            io.write(M, !IO),
            io.nl(!IO)
        else
            io.write_string("no report\n", !IO)
        ),
        reports(N - 1, Report, !IO)
    else
        true
    ).

% CHECK:got_high(2)
% CHECK-NEXT:got_low(1)
% CHECK-NEXT:got_low(3)
% CHECK-NEXT:got_tagged(70)
% CHECK-NEXT:timed_out
% CHECK-NEXT:got_tagged(80)
% CHECK-NEXT:got_low(6)
% CHECK-NEXT:got_low(5)
% CHECK-NEXT:stopped

% Five receives in selector, and one in loop, each a loop of the runtime's
% calls; nothing in the device's code is called through a pointer.
% LLVM-LABEL: define internal void @process_gpu_select.selector_5_0(
% LLVM-COUNT-5: call i64 @merk_device_select_next(
% LLVM-LABEL: define internal void @process_gpu_select.loop_4_0(
% LLVM: call i64 @merk_device_select_next(
% POINTER-NOT: call {{[^@]*}} %{{[0-9a-z._]+}}(
