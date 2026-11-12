% Flat records as GPU processes' messages and state (docs/INPROGRESS.md,
% Flat records, step 1). A mover is sent a particle, two points and a mass,
% and a number of steps, moves it by value and sends it back; a weigher
% keeps the sum of the particles it is sent as its state, a record argument
% of its loop. The program runs its kinds at the host here, and at the GPU
% in process-records-gpu.test. particle's two fields of one record type
% make it a record too, a struct of two structs.
% Temporary (docs/INPROGRESS.md, Flat records, step 2b): a record is boxed
% into a cell where it is sent, and the copies walk its type_info, until
% messages travel as their bytes.
% The -O0 build has no --offload-arch: AMDGPU's code generation at -O0
% stops on its musttail calls in divergent control flow (design document,
% "LLVM bugs to report").
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -emit-llvm -fprocess-declarations %s -o %t.ll
% RUN: FileCheck %s --check-prefix=LLVM < %t.ll
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fprocess-declarations %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: %libomp-env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 MERK_CHECK_MESSAGES=1 %t host 2>&1 | FileCheck %s --check-prefixes=CHECK,MESSAGES --match-full-lines --strict-whitespace
% RUN: %libomp-env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 MERK_CHECK_MESSAGES=1 %t2 host 2>&1 | FileCheck %s --check-prefixes=CHECK,MESSAGES --match-full-lines --strict-whitespace
%
% LLVM-LABEL: define internal tailcc void @process_records.move_f1_0(ptr %0, { { i32, i32 }, { i32, i32 }, double } %1) {
% LLVM-NOT: merk_alloc
% LLVM: ret void
% LLVM-LABEL: define internal tailcc void @process_records.weigh_5_0(ptr addrspace(1) %0, ptr addrspace(1) %1, { { i32, i32 }, { i32, i32 }, double } %2) {
%
% Mercury has no process module, so the output is written by hand: each step
% adds the velocity to the position, takes one from its y and halves the
% mass.
% CHECK:[message(particle(point(1i32, 2i32), point(3i32, 4i32), 64.0)), message(particle(point(10i32, 11i32), point(3i32, 1i32), 8.0)), message(particle(point(-4i32, -8i32), point(-1i32, -5i32), 2.0))]
% CHECK-NEXT:message(particle(point(55i32, 110i32), point(0i32, 0i32), 55.0))
% MESSAGES-NEXT:Mercury runtime: 0 messages are neither read nor freed

:- module process_records.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, int32, list, process, string.

:- type point ---> point(x :: int32, y :: int32).
:- type particle
    --->    particle(
                pos :: point,
                vel :: point,
                mass :: float
            ).

:- type work
    --->    step(particle, int)
    ;       stop.

    % Moves each particle it is sent the given number of steps, by value,
    % and sends it back.
:- process mover(mailbox(particle)) receives work on gpu.

mover(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(step(P, N)) then
        send(Reply, advance(N, P), !IO),
        mover(Reply, Self, !IO)
    else
        true
    ).

:- func advance(int, particle) = particle.
advance(N, P) =
    ( if N =< 0 then
        P
    else
        advance(N - 1, move(P))
    ).

:- func move(particle) = particle.
move(particle(point(X, Y), point(DX, DY), M)) =
    particle(point(X + DX, Y + DY), point(DX, DY - 1i32), M * 0.5).

    % Takes particles and answers with their total mass and the sum of their
    % positions, after done.
:- process weigher(mailbox(particle)) receives particle on gpu.

weigher(Reply, Self, !IO) :-
    weigh(Reply, Self, particle(point(0i32, 0i32), point(0i32, 0i32), 0.0),
        !IO).

:- pred weigh(mailbox(particle)::in, mailbox(particle)::in, particle::in,
    io::di, io::uo) is det.

weigh(Reply, Self, Sum, !IO) :-
    receive(Self, any, after(10000), R, !IO),
    ( if R = message(P) then
        ( if P ^ mass < 0.0 then
            send(Reply, Sum, !IO)
        else
            weigh(Reply, Self, add(Sum, P), !IO)
        )
    else
        true
    ).

:- func add(particle, particle) = particle.
add(A, B) =
    particle(point(A ^ pos ^ x + B ^ pos ^ x, A ^ pos ^ y + B ^ pos ^ y),
        A ^ vel, A ^ mass + B ^ mass).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Back, !IO),
    spawn(Place, mover(Back), M, !IO),
    P0 = particle(point(1i32, 2i32), point(3i32, 4i32), 64.0),
    send(M, step(P0, 0), !IO),
    send(M, step(P0, 3), !IO),
    send(M, step(P0 ^ vel := point(-1i32, 0i32), 5), !IO),
    send(M, stop, !IO),
    receive(Back, any, after(10000), R1, !IO),
    receive(Back, any, after(10000), R2, !IO),
    receive(Back, any, after(10000), R3, !IO),
    io.write_line([R1, R2, R3], !IO),
    spawn(Place, weigher(Back), W, !IO),
    list.foldl(send(W), list.map(make, 1 .. 10), !IO),
    send(W, particle(point(0i32, 0i32), point(0i32, 0i32), -1.0), !IO),
    receive(Back, any, after(10000), R4, !IO),
    io.write_line(R4, !IO).

:- func make(int) = particle.
make(I) = particle(point(int32.cast_from_int(I), int32.cast_from_int(2 * I)),
    point(0i32, 0i32), float(I)).
