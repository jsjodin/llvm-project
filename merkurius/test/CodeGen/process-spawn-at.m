% spawn(Place, Kind(Arguments), Mailbox, !IO) (docs/INPROGRESS.md, GPU
% processes without omp target, step 4c, decision D2): a process
% declaration's kind spawned at a place. At the host it runs as spawn/4's
% closure. At a GPU, a kind declared on gpu starts in its module's kernel,
% with its arguments' words, an int, a float and a character as one each and
% a mailbox as five, which process-spawn-at-gpu.test runs. A kind that runs
% on the host, or a GPU that does not exist, is a software error at the
% spawn, which names process.spawn. A closure in a variable, not a kind
% written out, runs at the host, and at a GPU is a software error. The program links LLVM's offload library, through which the
% runtime finds the GPUs. At -O0 its image links too, as its branches are
% uniform (Lowering.cpp, MakeProcessBranchesUniform).
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t0
% RUN: %libomp-env %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: not %libomp-env %t2 host-kind 2>&1 | FileCheck %s --check-prefix=HOST-KIND
% RUN: not %libomp-env %t2 no-gpu 2>&1 | FileCheck %s --check-prefix=NO-GPU
% RUN: %libomp-env %t2 closure-host | FileCheck %s --check-prefix=CLOSURE-HOST --match-full-lines
% RUN: not %libomp-env %t2 closure-gpu 2>&1 | FileCheck %s --check-prefix=CLOSURE-GPU

:- module process_spawn_at.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, float, int, list, process.

    % Adds up the numbers it receives until 0, then sends the sum, scaled,
    % with how far the character is past 'A', to Reply.
:- process summer(int, float, character, mailbox(int)) receives int on gpu.

summer(Sum, Scale, C, Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N \= 0 then
        summer(Sum + N, Scale, C, Reply, Self, !IO)
    else
        send(Reply, truncate_to_int(float(Sum) * Scale) +
            char.to_int(C) - char.to_int('A'), !IO)
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

    % Unifies through a type variable, which stops the program for closures.
:- pred same(T::in, T::in) is semidet.
:- pragma no_inline(same/2).

same(X, Y) :-
    X = Y.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    new_mailbox(Me, !IO),
    ( if Args = ["host-kind"] then
        spawn(gpu(0), doubler(Me), _, !IO)
    else if Args = ["no-gpu"] then
        spawn(gpu(1000000), summer(1, 0.5, 'A', Me), _, !IO)
    else if Args = ["closure-host"] then
        % A closure, not a kind written out, runs at the host.
        P = summer(1, 0.5, 'B', Me),
        spawn(host, P, S, !IO),
        send(S, 3, !IO),
        send(S, 0, !IO),
        receive(Me, any, infinity, R, !IO),
        ( if R = message(N) then
            io.write_int(N, !IO),
            io.nl(!IO)
        else
            true
        )
    else if Args = ["closure-gpu"] then
        P = summer(1, 0.5, 'A', Me),
        spawn(gpu(0), P, _, !IO)
    else if Args = ["closures"] then
        % The runtime stops the program while the process waits.
        spawn(gpu(0), summer(1, 0.5, 'A', Me), _, !IO),
        P = (pred(X::out) is det :- X = 1),
        ( if same(P, P) then
            io.write_string("same\n", !IO)
        else
            true
        )
    else if Args = ["leave"] then
        % main ends while the process waits for more.
        spawn(gpu(0), summer(1, 0.5, 'A', Me), S, !IO),
        send(S, 5, !IO),
        io.write_string("main ends\n", !IO)
    else
        Place = ( if Args = ["gpu"] then gpu(0) else host ),
        spawn(Place, summer(1, 0.5, 'C', Me), S, !IO),
        send(S, 20, !IO),
        send(S, 63, !IO),
        send(S, 0, !IO),
        receive(Me, any, after(10000), R1, !IO),
        ( if R1 = message(N1) then
            io.write_int(N1, !IO), io.nl(!IO),
            spawn(host, doubler(Me), D, !IO),
            send(D, N1, !IO),
            receive(Me, any, infinity, R2, !IO),
            ( if R2 = message(N2) then
                io.write_int(N2, !IO), io.nl(!IO)
            else
                true
            )
        else
            io.write_string("no answer\n", !IO)
        )
    ).
% CHECK:44
% CHECK-NEXT:88
% HOST-KIND:Software Error: process.spawn: the process kind doubler runs on the host, not at gpu(0): its declaration does not say on gpu
% CLOSURE-HOST:3
% CLOSURE-GPU:Software Error: process.spawn: a closure cannot run at gpu(0): write the process kind with its arguments in the call, as in spawn(gpu(0), kind(Arguments), Mailbox, !IO)
% NO-GPU:Software Error: process.spawn: there is no GPU 1000000: this node has {{.*}}
