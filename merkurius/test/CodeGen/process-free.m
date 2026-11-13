% merk_device_free (runtime/device/device.cpp), which merk.free calls in a
% GPU process (docs/INPROGRESS.md, Memory that a GPU process gives back),
% called here by a foreign_proc, as no pass frees yet. A freer makes an
% array of each size it is asked, its indices, answers with their sum and a
% string literal's length, and then frees the array and the string: the
% array's block goes back to the cache of large blocks, for the next array,
% and the string is left alone, as it is in the image's merk_static section.
% At the host, which has a collector, the C code frees nothing. The kind
% runs at the host here, and on the GPU in process-free-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand: each sum is N * (N - 1) / 2 + 6.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_free.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- pragma foreign_decl("C", local, "
#ifdef __AMDGPU__
void merk_device_free(void *);
#endif
").

:- pred free_array(array(int)::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    free_array(A::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
#ifdef __AMDGPU__
    merk_device_free((void *)A);
#endif
").

:- pred free_string(string::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    free_string(S::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
#ifdef __AMDGPU__
    merk_device_free(S);
#endif
").

    % Answers each size with the sum of an array of that many indices and
    % the length of "static", until a size of 0.
:- process freer(mailbox(int)) receives int on gpu.

freer(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N > 0 then
        array.init(N, 0, A0),
        fill(0, N, A0, A),
        S = "static",
        send(Reply, sum(A, 0, N, 0) + string.length(S), !IO),
        free_array(A, !IO),
        free_string(S, !IO),
        freer(Reply, Self, !IO)
    else
        true
    ).

:- pred fill(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
fill(I, N, !A) :-
    ( if I < N then
        !A ^ elem(I) := I,
        fill(I + 1, N, !A)
    else
        true
    ).

:- func sum(array(int), int, int, int) = int.
sum(A, I, N, S) = ( if I < N then sum(A, I + 1, N, S + A ^ elem(I)) else S ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    new_mailbox(Me, !IO),
    new_mailbox(Downs, !IO),
    spawn(Place, freer(Me), F, !IO),
    monitor(F, Downs, id, _, !IO),
    list.foldl(ask(F, Me), [200000, 200000, 10, 200000, 200000], !IO),
    send(F, 0, !IO),
    receive(Downs, any, after(60000), _, !IO).

:- func id(down_info) = down_info.
id(D) = D.

:- pred ask(mailbox(int)::in, mailbox(int)::in, int::in, io::di, io::uo)
    is det.
ask(F, Me, N, !IO) :-
    send(F, N, !IO),
    receive(Me, any, after(60000), R, !IO),
    ( if R = message(Sum), Sum = N * (N - 1) / 2 + 6 then
        io.format("%d: right\n", [i(N)], !IO)
    else
        io.format("%d: wrong\n", [i(N)], !IO)
    ).

% CHECK:200000: right
% CHECK-NEXT:200000: right
% CHECK-NEXT:10: right
% CHECK-NEXT:200000: right
% CHECK-NEXT:200000: right
