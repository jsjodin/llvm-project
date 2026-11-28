% DESIGN: Sized floating point types
% float32s in messages to and from GPU processes: a scaler receives a
% float32 alone, an array of float32s, packed 4 bytes to an element, and a
% record with float32 fields, and answers each with float32s it computes,
% the array's into a new one, with float32's operators and math_float32's sqrt, which on the GPU are
% the device's single-precision instructions and its library's sqrtf. Run
% with no argument the scaler runs at the host; run-float32-processes-
% gpu.test runs it on the GPU and checks the same output.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/float32_processes_translation.c
%
% RUN: merkc -O0 -fprocess-declarations %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:message(half(0.05f32))
% OUTPUT-NEXT:message(scaled(0.2505f32, array([0.5f32, 1.25f32, -1.5f32, 0.0005f32])))
% OUTPUT-NEXT:message(length(5.0f32, 7))
% OUTPUT-NEXT:message(length(1.4142135f32, 8))
% OUTPUT-NOT:{{.}}
:- module run_float32_processes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, float32, int, list, math_float32, process.

:- type vec
    --->    vec(x :: float32, y :: float32, tag :: int).

:- type request
    --->    halve(float32)
    ;       scale(float32, array(float32))
    ;       norm(vec)
    ;       stop.

:- type answer
    --->    half(float32)
    ;       scaled(float32, array(float32))     % the sum and the array
    ;       length(float32, int).

:- process scaler(mailbox(answer)) receives request on gpu.

scaler(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(Request) then
        (
            Request = halve(X),
            send(Reply, half(X / 2.0f32), !IO),
            scaler(Reply, Self, !IO)
        ;
            Request = scale(K, A),
            scale_from(0, K, A, array.init(array.size(A), 0.0f32), B,
                0.0f32, Sum),
            send(Reply, scaled(Sum, B), !IO),
            scaler(Reply, Self, !IO)
        ;
            Request = norm(vec(X, Y, Tag)),
            send(Reply, length(sqrt(X * X + Y * Y), Tag), !IO),
            scaler(Reply, Self, !IO)
        ;
            Request = stop
        )
    else
        true
    ).

    % Sets each element of B from I to K times A's, and sums them.
:- pred scale_from(int::in, float32::in, array(float32)::in,
    array(float32)::array_di, array(float32)::array_uo,
    float32::in, float32::out) is det.

scale_from(I, K, A, !B, !Sum) :-
    ( if I < array.size(A) then
        E = K * array.lookup(A, I),
        array.set(I, E, !B),
        !:Sum = !.Sum + E,
        scale_from(I + 1, K, A, !B, !Sum)
    else
        true
    ).

:- pred report(mailbox(answer)::in, io::di, io::uo) is det.

report(Box, !IO) :-
    receive(Box, any, after(10000), R, !IO),
    io.write(R, !IO),
    io.nl(!IO).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Answers, !IO),
    spawn(Place, scaler(Answers), S, !IO),
    send(S, halve(0.1f32), !IO),
    report(Answers, !IO),
    send(S, scale(0.5f32, array([1.0f32, 2.5f32, -3.0f32, 1.0e-3f32])), !IO),
    report(Answers, !IO),
    send(S, norm(vec(3.0f32, 4.0f32, 7)), !IO),
    report(Answers, !IO),
    send(S, norm(vec(1.0f32, 1.0f32, 8)), !IO),
    report(Answers, !IO),
    send(S, stop, !IO).
