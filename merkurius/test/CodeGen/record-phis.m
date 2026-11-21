% Flat records through branches split into their fields (lib/CodeGen/
% RecordPhis.h): wave adds a wave to one of two vec3s, as the shallow-water
% port's solver does, and on a GPU, where a procedure's outputs come back as
% a struct, the inliner joins wave's two ends in a phi of the whole record.
% Split, each field is a phi of its own, and a field nothing reads is gone:
% right_first keeps only P's first. pick's select of a nested record becomes
% a select of each of its four numbers, on the host too. On the host the
% outputs go through memory, which SROA splits already.
%
% RUN: merkc -fc1 -O2 -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-llvm %s -o %t-device.ll
% RUN: FileCheck %s --check-prefix=DEVICE < %t-device.ll
% RUN: not grep -E '= (phi|select i1 [^,]*,) [{[]' %t-device.ll
% RUN: merkc -fc1 -O2 -fprocess-declarations -emit-llvm %s -o %t.ll
% RUN: FileCheck %s --check-prefix=LLVM < %t.ll
% RUN: not grep -E '= (phi|select i1 [^,]*,) [{[]' %t.ll
% RUN: merkc -O2 -fprocess-declarations %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fprocess-declarations %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
%
% DEVICE-LABEL: define internal fastcc { double, double, double } @record_phis.waves_4_0(
% DEVICE: phi double
% DEVICE-LABEL: define internal fastcc double @record_phis.right_first_f3_0(
% DEVICE: %.pn3.f0.{{.*}} = phi double
% DEVICE-NOT: .f1.
% DEVICE: ret double
%
% LLVM-LABEL: define internal tailcc void @record_phis.pick_f3_0(
% LLVM: select i1 %{{.*}}, double
% LLVM: select i1 %{{.*}}, double
% LLVM: select i1 %{{.*}}, double
% LLVM: select i1 %{{.*}}, double
% LLVM: ret void
%
% mmc has no process module: the first three lines, the process's, are
% written by hand; the rest are mmc's.
% CHECK:17.0000 2.0000
% CHECK-NEXT:6.0000 0.0000
% CHECK-NEXT:5.7500 2.2500
% CHECK-NEXT:vec3(2.0, 1.0, -0.5)
% CHECK-NEXT:vec3(0.0, 0.75, 1.5)
% CHECK-NEXT:vec3(0.0, -6.0, -3.0)
% CHECK-NEXT:cell(vec3(2.0, 1.0, -0.5), 1.0)
% CHECK-NEXT:cell(vec3(0.0, 0.75, 1.5), 2.0)
% CHECK-NEXT:1.5
% CHECK-NEXT:17.0
% CHECK-NEXT:6.0
% CHECK-NEXT:vec3(4.0, -13.0, -10.0)

:- module record_phis.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list, process, string.

:- type vec3
    --->    vec3(float, float, float).

:- type cell
    --->    cell(vec3, float).

:- type normal
    --->    normal(float, float).

:- func vec3 + vec3 = vec3.
:- pragma inline((+)/2).
vec3(A0, A1, A2) + vec3(B0, B1, B2) = vec3(A0 + B0, A1 + B1, A2 + B2).

:- func float * vec3 = vec3.
:- pragma inline((*)/2).
K * vec3(E0, E1, E2) = vec3(K * E0, K * E1, K * E2).

:- pred wave(float::in, float::in, vec3::in, vec3::in, vec3::out,
    vec3::in, vec3::out) is det.
:- pragma inline(wave/7).
wave(L, K, E, !P, !M) :-
    ( if L > 0.0 then
        !:P = !.P + K * E
    else
        !:M = !.M + K * E
    ).

:- pred waves(float::in, float::in, vec3::out, vec3::out) is det.
:- pragma no_inline(waves/4).
waves(L0, L1, P, M) :-
    E0 = vec3(1.0, L0, L1),
    E1 = vec3(0.0, -L1, L0),
    some [!P, !M] (
        !:P = vec3(0.0, 0.0, 0.0),
        !:M = vec3(0.0, 0.0, 0.0),
        wave(L0, 2.0, E0, !P, !M),
        wave(L1, 3.0, E1, !P, !M),
        P = !.P,
        M = !.M
    ).

:- func right(float, float) = vec3.
:- pragma no_inline(right/2).
right(L0, L1) = P :-
    waves(L0, L1, P, _).

:- func pick(int, cell, cell) = cell.
:- pragma no_inline(pick/3).
pick(I, A, B) = ( if I > 0 then A else B ).

:- func normal_of(float) = normal.
:- pragma inline(normal_of/1).
normal_of(X) = ( if X > 0.0 then normal(X, X * X) else normal(0.0, 0.0) ).

:- func first(float) = float.
:- pragma no_inline(first/1).
first(X) = Y :-
    normal_of(X) = normal(Y, _).

:- pred wave2(float::in, float::in, normal::in, normal::in, normal::out,
    normal::in, normal::out) is det.
:- pragma inline(wave2/7).
wave2(L, K, normal(E0, E1), !P, !M) :-
    ( if L > 0.0 then
        !.P = normal(P0, P1),
        !:P = normal(P0 + K * E0, P1 + K * E1)
    else
        !.M = normal(M0, M1),
        !:M = normal(M0 + K * E0, M1 + K * E1)
    ).

    % Only the right side's first field: the rest is dead once the
    % records through the branches are split.
:- func right_first(float, float, float) = float.
:- pragma no_inline(right_first/3).
right_first(L0, L1, L2) = R :-
    some [!P, !M] (
        !:P = normal(0.0, 0.0),
        !:M = normal(0.0, 0.0),
        wave2(L0, 2.0, normal(1.0, L0), !P, !M),
        wave2(L1, 3.0, normal(L1, 1.0), !P, !M),
        wave2(L2, 5.0, normal(L2, L0), !P, !.M, _),
        !.P = normal(R, _)
    ).

    % The same, in a GPU process: it answers each L with right_first's and
    % the largest field of right's.
:- process solver(mailbox(float)) receives float on gpu.
solver(Reply, Self, !IO) :-
    receive(Self, any, after(10000), R, !IO),
    ( if R = message(L) then
        send(Reply, right_first(L, -2.0 * L, 3.0 * L), !IO),
        right(L, 1.0 - L) = vec3(X, Y, Z),
        send(Reply, float.max(X, float.max(Y, Z)), !IO),
        solver(Reply, Self, !IO)
    else
        true
    ).

:- func sum(list(float), vec3) = vec3.
sum([], V) = V.
sum([L | Ls], V) = sum(Ls, V + right(L, 1.0 - L)).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Back, !IO),
    spawn(Place, solver(Back), S, !IO),
    list.foldl(ask(S, Back), [1.0, -1.0, 0.25], !IO),
    waves(0.5, -0.25, P, M),
    io.write(P, !IO), io.nl(!IO),
    io.write(M, !IO), io.nl(!IO),
    io.write(right(-1.0, 2.0), !IO), io.nl(!IO),
    io.write(pick(1, cell(P, 1.0), cell(M, 2.0)), !IO), io.nl(!IO),
    io.write(pick(0, cell(P, 1.0), cell(M, 2.0)), !IO), io.nl(!IO),
    io.write(first(1.5) + first(-2.0), !IO), io.nl(!IO),
    io.write(right_first(1.0, -2.0, 3.0), !IO), io.nl(!IO),
    io.write(right_first(-1.0, 2.0, -3.0), !IO), io.nl(!IO),
    io.write(sum([0.5, -0.5, 2.0, -3.0], vec3(0.0, 0.0, 0.0)), !IO), io.nl(!IO).

:- pred ask(mailbox(float)::in, mailbox(float)::in, float::in,
    io::di, io::uo) is det.
ask(S, Back, L, !IO) :-
    send(S, L, !IO),
    receive(Back, any, after(10000), R1, !IO),
    receive(Back, any, after(10000), R2, !IO),
    ( if R1 = message(A), R2 = message(B) then
        io.format("%.4f %.4f\n", [f(A), f(B)], !IO)
    else
        io.write_string("no answer\n", !IO)
    ).
