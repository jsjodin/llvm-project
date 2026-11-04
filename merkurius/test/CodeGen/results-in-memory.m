% A procedure whose outputs, after its success flag if it can fail, need
% more registers than the target returns results in (TargetInfo::Results)
% returns them in memory, through a pointer before its inputs (design
% document, "Calling convention"). Its tail calls pass the pointer they
% were given on, since the callee's outputs are the caller's, and stay
% musttail; a back end that returns results in memory itself cannot make
% such a call. Any other call gives a slot of the caller's and loads the
% outputs from it. On x86-64, three words and two floats fit, so four ints,
% three ints after a success flag or three floats do not; on AArch64,
% eight of each fit. The loops run ten million times, in constant stack
% space.
%
% RUN: merkc -fc1 -fdo-loops -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=X86
% RUN: merkc -fc1 -fdo-loops -emit-llvm -triple aarch64-unknown-linux-gnu %s | FileCheck %s --check-prefix=AARCH64
% RUN: merkc -fc1 -fdo-loops -emit-obj -triple aarch64-unknown-linux-gnu %s -o /dev/null
% RUN: merkc -fdo-loops %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module results_in_memory.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

main(!IO) :-
    four(10000000, 0, 0, 0, 0, A, B, C, D),
    io.write_int(A + B + C + D, !IO), io.nl(!IO),
    ( if three_or_fail(10000000, 0, 0, 0, X, Y, Z) then
        io.write_int(X + Y + Z, !IO)
    else
        io.write_string("failed", !IO)
    ),
    io.nl(!IO),
    ( if three_or_fail(-1, 0, 0, 0, _, _, _) then
        io.write_string("succeeded", !IO)
    else
        io.write_string("failed", !IO)
    ),
    io.nl(!IO),
    floats(10000000, 0.0, 0.0, 0.0, F, G, H),
    io.write_float(F + G + H, !IO), io.nl(!IO),
    ping(10000000, 0, 0, 0, P, Q, R, S),
    io.write_int(P + Q + R + S, !IO), io.nl(!IO),
    array.init(10, 0, A0),
    loop(A0, A1, Lo, Hi, Sum),
    io.write_int(A1 ^ elem(9) + Lo + Hi + Sum, !IO), io.nl(!IO).
% OUTPUT:40000000
% OUTPUT:30000000
% OUTPUT:failed
% OUTPUT:30000000.0
% OUTPUT:30000000
% OUTPUT:63

% X86-LABEL: define internal tailcc void @results_in_memory.four_9_0(ptr %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5) {
% X86:         %[[OUT:.*]] = getelementptr { i64, i64, i64, i64 }, ptr %0, i32 0, i32 3
% X86-NEXT:    store i64 %{{.*}}, ptr %[[OUT]]
% X86-NEXT:    ret void
% X86:         musttail call tailcc void @results_in_memory.four_9_0(ptr %0,
% X86-NEXT:    ret void
% AARCH64-LABEL: define internal tailcc { i64, i64, i64, i64 } @results_in_memory.four_9_0(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4) {
:- pred four(int::in, int::in, int::in, int::in, int::in,
    int::out, int::out, int::out, int::out) is det.
four(N, A0, B0, C0, D0, A, B, C, D) :-
    ( if N = 0 then
        A = A0, B = B0, C = C0, D = D0
    else
        four(N - 1, A0 + 1, B0 + 1, C0 + 1, D0 + 1, A, B, C, D)
    ).

% The flag is returned, in a register; failing stores nothing.
% X86-LABEL: define internal tailcc i1 @results_in_memory.three_or_fail_7_0(ptr %0, i64 %1, i64 %2, i64 %3, i64 %4) {
% X86:         musttail call tailcc i1 @results_in_memory.three_or_fail_7_0(ptr %0,
% X86:         ret i1 false
% AARCH64-LABEL: define internal tailcc { i1, i64, i64, i64 } @results_in_memory.three_or_fail_7_0(
:- pred three_or_fail(int::in, int::in, int::in, int::in,
    int::out, int::out, int::out) is semidet.
three_or_fail(N, X0, Y0, Z0, X, Y, Z) :-
    N >= 0,
    ( if N = 0 then
        X = X0, Y = Y0, Z = Z0
    else
        three_or_fail(N - 1, X0 + 1, Y0 + 1, Z0 + 1, X, Y, Z)
    ).

% X86-LABEL: define internal tailcc void @results_in_memory.floats_7_0(ptr %0, i64 %1, double %2, double %3, double %4) {
:- pred floats(int::in, float::in, float::in, float::in,
    float::out, float::out, float::out) is det.
floats(N, F0, G0, H0, F, G, H) :-
    ( if N = 0 then
        F = F0, G = G0, H = H0
    else
        floats(N - 1, F0 + 1.0, G0 + 1.0, H0 + 1.0, F, G, H)
    ).

% Tail calls between two procedures pass the pointer on too, with no slot.
% X86-LABEL: define internal tailcc void @results_in_memory.ping_8_0(ptr %0,
% X86-NOT:     alloca
% X86:         musttail call tailcc void @results_in_memory.pong_8_0(ptr %0,
:- pred ping(int::in, int::in, int::in, int::in,
    int::out, int::out, int::out, int::out) is det.
ping(N, A0, B0, C0, A, B, C, D) :-
    ( if N = 0 then
        A = A0, B = B0, C = C0, D = N
    else
        pong(N - 1, A0 + 1, B0 + 1, C0 + 1, A, B, C, D)
    ).
:- pred pong(int::in, int::in, int::in, int::in,
    int::out, int::out, int::out, int::out) is det.
pong(N, A0, B0, C0, A, B, C, D) :-
    ( if N = 0 then
        A = A0, B = B0, C = C0, D = N
    else
        ping(N - 1, A0 + 1, B0 + 1, C0 + 1, A, B, C, D)
    ).

% The loop that found it: an array and three ints. Its call is not a tail
% call: the loop's caller gives a slot, which it loads.
% X86-LABEL: define internal tailcc void @results_in_memory.loop_5_0(ptr %0, ptr %1) {
% X86:         %[[SLOT:.*]] = alloca { i64, i64, i64, ptr }
% X86:         call tailcc void @results_in_memory.loop__loop_{{.*}}(ptr %[[SLOT]],
% X86-NEXT:    %[[FIRST:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[SLOT]], i32 0, i32 0
% X86-NEXT:    load i64, ptr %[[FIRST]]
:- pred loop(array(int)::array_di, array(int)::array_uo,
    int::out, int::out, int::out) is det.
loop(!A, Lo, Hi, Sum) :-
    ( for(I, 0, 9), fromto(max_int, L0, L1, Lo), fromto(min_int, H0, H1, Hi),
        fromto(0, S0, S1, Sum) do
        !A ^ elem(I) := I,
        L1 = int.min(L0, I),
        H1 = int.max(H0, I),
        S1 = S0 + I
    ).
