% Group-private arrays (docs/INPROGRESS.md, GPU processes that run loops,
% (s5)): a teams distribute loop's dyn_groupprivate clause names an array
% that the loop's body makes with array.init, which each team has in its
% local memory for the iteration. The worker fills its tile's array G0 by
% one loop across the team's threads and reads it back reversed by
% another; one of 20000 ints does not fit in a team's local memory, so it
% is in the process's memory instead (fallback(default_mem), the default).
%
% mmc has no process module to run it with, so the output is written by
% hand (process-group-private-gpu.test runs it).
%
% The device's compile marks the array and starts each iteration of the
% body by ending the last one's arrays.
% RUN: merkc -fc1 -fprocess-loops -fopenmp-version=61 -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o - | FileCheck %s --check-prefix=MERK
% MERK-LABEL: merk.proc private @process_group_private.tiles__body_{{[0-9_]+}}(
% MERK-NEXT: merk.foreign_call @merk_device_group_begin() : () -> ()
% MERK: merk.array_new %{{.*}} {merk.group_private = 1 : i32} -> <i64>
%
% FreeAtLastUse frees no group-private array, nor fill's result, the same
% array updated in place, which the runtime frees at the iteration's end.
% RUN: merkc -fc1 -fprocess-loops -fopenmp-version=61 -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -fdebug-dump-frees %s -o - | FileCheck %s --check-prefix=FREES
% FREES-LABEL: merk.proc private @process_group_private.tiles__body_{{[0-9_]+}}(
% FREES: %[[G:[0-9]+]] = merk.array_new {{.*}}merk.group_private
% FREES-NEXT: merk.call @process_group_private.fill_{{[0-9_]+}}(%{{.*}}, %{{.*}}, %[[G]])
% FREES-NEXT: merk.call @process_group_private.reverse_into_{{[0-9_]+}}(
% FREES-NEXT: merk.succeed
%
% Lowering asks the runtime for the next array in group memory, and marks
% the image as one whose kernel needs local memory for them, over the
% device runtime's weak 0, protected so that the host can read it.
% RUN: merkc -fc1 -fprocess-loops -fopenmp-version=61 -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
% LLVM: @merk_device_group_private = protected global i32 1, align 4
% LLVM-LABEL: define internal ptr @process_group_private.tiles__body_{{[0-9_]+}}(
% LLVM-NEXT: call void @merk_device_group_begin()
% LLVM: call void @merk_device_group_next(i32 1)
% LLVM-NEXT: call ptr @merk_array_new(
%
% On the host, as in a kind spawned there, the clause changes nothing.
% RUN: merkc -fc1 -fopenmp -fopenmp-version=61 -fdo-loops -fprocess-declarations -triple x86_64-unknown-linux-gnu -emit-merk %s -o - | FileCheck %s --check-prefix=HOST
% HOST-NOT: merk.group_private
% HOST-NOT: merk_device_group

:- module process_group_private.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- process worker(mailbox(int)) receives int on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(T), T > 0 then
        Tiles = 50,
        array.init(Tiles * T, 0, Out0),
        tiles(Tiles, T, Out0, Out),
        ( for(I, 0, Tiles * T - 1), fromto(0, S0, S1, Sum) do
            S1 = S0 + Out ^ unsafe_elem(I)
        ),
        send(Reply, Sum, !IO),
        worker(Reply, Self, !IO)
    else
        true
    ).

    % Each tile's T elements, through a group-private array G: filled in
    % order, and read back reversed.
:- pred tiles(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
tiles(Tiles, T, !Out) :-
    %$omp teams distribute dyn_groupprivate(G0)
    ( for(J, 0, Tiles - 1) do
        array.init(T, 0, G0),
        fill(J, T, G0, G),
        reverse_into(J, T, G, !Out)
    ).

:- pred fill(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
fill(J, T, !G) :-
    %$omp parallel do
    ( for(K, 0, T - 1) do
        !G ^ unsafe_elem(K) := J * 1000 + K
    ).

:- pred reverse_into(int::in, int::in, array(int)::array_ui,
    array(int)::array_di, array(int)::array_uo) is det.
reverse_into(J, T, G, !Out) :-
    %$omp parallel do
    ( for(K, 0, T - 1) do
        !Out ^ unsafe_elem(J * T + K) := G ^ unsafe_elem(T - 1 - K) * (K + 1)
    ).

main(!IO) :-
    new_mailbox(Me, !IO),
    spawn(gpu(0), worker(Me), W, !IO),
    list.foldl(run(Me, W), [16, 300, 5000, 20000], !IO),
    send(W, 0, !IO).
% CHECK:16: 166634000
% CHECK-NEXT:300: 55533747500
% CHECK-NEXT:5000: 16357229125000
% CHECK-NEXT:20000: 311678916500000

:- pred run(mailbox(int)::in, mailbox(int)::in, int::in, io::di, io::uo)
    is det.
run(Me, W, T, !IO) :-
    send(W, T, !IO),
    receive(Me, any, after(20000), R, !IO),
    ( if R = message(S) then
        io.format("%d: %d\n", [i(T), i(S)], !IO)
    else
        io.write_string("no answer\n", !IO)
    ).
