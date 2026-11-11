% Every team of a GPU processes' kernel runs processes (docs/INPROGRESS.md,
% GPU processes without omp target, step 4b): seventeen adders, all spawned
% before any is sent a number, fill more slots than a team has, and each
% answers. The kernel's teams must all be resident at once, as a slot waits
% for a process only once its team runs. The device runtime's Fail trapped,
% which made the kernel's stack dynamic, and the GPU then held back all its
% teams but the first, as it did not hold their scratch: the adders in the
% others never answered. Now the host also waits for the teams to start
% when it launches the kernel (process-teams-gpu.test).
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target
% RUN: rm -rf %t.dir && mkdir %t.dir
% RUN: merkc -c -fprocess-declarations --offload-arch=gfx1030 -save-temps=obj %s -o %t.dir/p.o
% RUN: llvm-readelf --notes %t.dir/process-teams-gfx1030-processes.elf | FileCheck %s --check-prefix=STACK
% STACK: .uses_dynamic_stack: false
% RUN: merkc -c -fprocess-declarations --offload-arch=gfx1030 -save-temps=obj %S/Inputs/process_teams_recursive.m -o %t.dir/r.o
% RUN: llvm-readelf --notes %t.dir/process_teams_recursive-gfx1030-processes.elf | FileCheck %s --check-prefix=DYNAMIC
% DYNAMIC: .uses_dynamic_stack: true

:- module process_teams.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

    % Sends Reply the number it receives, plus one, and ends.
:- process adder(mailbox(int)) receives int on gpu.

adder(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then send(Reply, N + 1, !IO) else true ).

main(!IO) :-
    new_mailbox(Me, !IO),
    Numbers = 1 .. 17,
    list.map_foldl(spawn_adder(Me), Numbers, Adders, !IO),
    send_numbers(Adders, Numbers, !IO),
    sum_answers(Me, list.length(Numbers), 0, Count, 0, Sum, !IO),
    io.format("%d answers, %d in all\n", [i(Count), i(Sum)], !IO).
% CHECK:17 answers, 170 in all

:- pred spawn_adder(mailbox(int)::in, int::in, mailbox(int)::out,
    io::di, io::uo) is det.

spawn_adder(Me, _, A, !IO) :-
    spawn(gpu(0), adder(Me), A, !IO).

:- pred send_numbers(list(mailbox(int))::in, list(int)::in,
    io::di, io::uo) is det.

send_numbers(As, Ns, !IO) :-
    ( if As = [A | As1], Ns = [N | Ns1] then
        send(A, N, !IO),
        send_numbers(As1, Ns1, !IO)
    else
        true
    ).

    % Receives Left more answers, each within ten seconds.
:- pred sum_answers(mailbox(int)::in, int::in, int::in, int::out, int::in,
    int::out, io::di, io::uo) is det.

sum_answers(Me, Left, !Count, !Sum, !IO) :-
    ( if Left = 0 then
        true
    else
        receive(Me, any, after(10000), R, !IO),
        ( if R = message(M) then
            !:Count = !.Count + 1,
            !:Sum = !.Sum + M,
            sum_answers(Me, Left - 1, !Count, !Sum, !IO)
        else
            true
        )
    ).
