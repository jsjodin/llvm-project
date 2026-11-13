% What a GPU process's blocks cost (docs/INPROGRESS.md, Memory that a GPU
% process gives back, f4), beside a message's round trip. A process on the
% GPU runs each measurement's rounds and answers when they are done; the
% host times the measurement with clock_gettime, less one round trip
% (README.md).
%
%   frees gpu|host alloc|malloc ROUNDS BYTES
%   frees gpu|host cold-alloc|cold-malloc ROUNDS BYTES
%   frees gpu|host array|none|ping ROUNDS [WORDS]

:- module frees.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

:- type task
    --->    alloc(int, int)
    ;       malloc(int, int)
    ;       cold_alloc(int, int)
    ;       cold_malloc(int, int)
    ;       array(int, int)
    ;       none(int)
    ;       ping
    ;       stop.

:- pragma foreign_decl("C", local, "
#include <stdint.h>
#include <stdlib.h>
#ifdef __AMDGPU__
void *merk_device_alloc(uint64_t);
void merk_device_free(void *);
#define BLOCK_ALLOC merk_device_alloc
#define BLOCK_FREE merk_device_free
#else
#define BLOCK_ALLOC malloc
#define BLOCK_FREE free
#endif
#define LARGE_STEP ((uint64_t)2 << 20)
").

    % Allocates a block of the bytes and frees it, each round, with the
    % runtime's merk_device_alloc and merk_device_free, as a GPU process's
    % cells and arrays are (runtime/device/device.cpp, NewBlock), or with
    % libc's malloc and free alone. A cold round's block is LARGE_STEP
    % bigger than the last, so that no cache has one.
:- pred blocks(int::in, int::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    blocks(Kind::in, Rounds::in, Bytes::in, Sum::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    uint64_t sum = 0;
    for (MR_Integer i = 0; i < Rounds; ++i) {
        uint64_t bytes = (uint64_t)Bytes + (Kind >= 2 ? i * LARGE_STEP : 0);
        volatile char *p = (Kind & 1) ? malloc(bytes) : BLOCK_ALLOC(bytes);
        p[0] = (char)i;
        sum += (uint64_t)p[0];
        if (Kind & 1) {
            free((void *)p);
        } else {
            BLOCK_FREE((void *)p);
        }
    }
    Sum = (MR_Integer)sum;
").

:- process allocator(mailbox(int)) receives task on gpu.

allocator(Host, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(Task), Task \= stop then
        run(Task, Sum, !IO),
        send(Host, Sum, !IO),
        allocator(Host, Self, !IO)
    else
        true
    ).

:- pred run(task::in, int::out, io::di, io::uo) is det.
run(alloc(Rounds, Bytes), Sum, !IO) :- blocks(0, Rounds, Bytes, Sum, !IO).
run(malloc(Rounds, Bytes), Sum, !IO) :- blocks(1, Rounds, Bytes, Sum, !IO).
run(cold_alloc(Rounds, Bytes), Sum, !IO) :- blocks(2, Rounds, Bytes, Sum, !IO).
run(cold_malloc(Rounds, Bytes), Sum, !IO) :- blocks(3, Rounds, Bytes, Sum, !IO).
run(array(Rounds, Words), arrays(Rounds, Words, 0), !IO).
run(none(Rounds), nothing(Rounds, 0), !IO).
run(ping, 0, !IO).
run(stop, 0, !IO).

    % Makes an array of the words each round, which is freed at its last
    % use (Frees.cpp), before the next round.
:- func arrays(int, int, int) = int.
arrays(Rounds, Words, Sum) =
    ( if Rounds > 0 then
        arrays(Rounds - 1, Words, Sum + touch(Rounds, Words))
    else
        Sum
    ).

:- func touch(int, int) = int.
touch(Round, Words) = A ^ elem(0) + array.size(A) :-
    array.init(Words, 0, A0),
    A = A0 ^ elem(0) := Round.

:- func nothing(int, int) = int.
nothing(Rounds, Sum) =
    ( if Rounds > 0 then nothing(Rounds - 1, Sum + Rounds) else Sum ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if
        Args = [Where, Name, RoundsText | Rest],
        string.to_int(RoundsText, Rounds),
        Size = ( if Rest = [S], string.to_int(S, N) then N else 0 ),
        task(Name, Rounds, Size, Task)
    then
        Place = ( if Where = "host" then host else gpu(0) ),
        new_mailbox(Me, !IO),
        new_mailbox(Downs, !IO),
        spawn(Place, allocator(Me), P, !IO),
        monitor(P, Downs, id, _, !IO),
        % Warms up the process and the path of a message.
        round_trips(P, Me, 100, !IO),
        now(T0, !IO),
        round_trips(P, Me, 1000, !IO),
        now(T1, !IO),
        Trip = (T1 - T0) / 1000,
        ( if Name = "ping" then
            now(T2, !IO),
            round_trips(P, Me, Rounds, !IO),
            now(T3, !IO),
            Time = T3 - T2
        else
            now(T2, !IO),
            send(P, Task, !IO),
            receive(Me, any, infinity, _, !IO),
            now(T3, !IO),
            Time = T3 - T2 - Trip
        ),
        io.format("%s %d rounds of %d: %.3f us a round\n",
            [s(Name), i(Rounds), i(Size),
                f(float(Time) / float(Rounds) / 1000.0)], !IO),
        send(P, stop, !IO),
        receive(Downs, any, after(60000), _, !IO)
    else
        io.write_string("usage: frees gpu|host KIND ROUNDS [SIZE]\n", !IO),
        io.set_exit_status(2, !IO)
    ).

:- pred task(string::in, int::in, int::in, task::out) is semidet.
task("alloc", R, S, alloc(R, S)).
task("malloc", R, S, malloc(R, S)).
task("cold-alloc", R, S, cold_alloc(R, S)).
task("cold-malloc", R, S, cold_malloc(R, S)).
task("array", R, S, array(R, S)).
task("none", R, _, none(R)).
task("ping", _, _, ping).

:- func id(down_info) = down_info.
id(D) = D.

:- pred round_trips(mailbox(task)::in, mailbox(int)::in, int::in,
    io::di, io::uo) is det.
round_trips(P, Me, N, !IO) :-
    ( if N > 0 then
        send(P, ping, !IO),
        receive(Me, any, infinity, _, !IO),
        round_trips(P, Me, N - 1, !IO)
    else
        true
    ).

:- pragma foreign_decl("C", "#include <time.h>").

:- pred now(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(Nanoseconds::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    Nanoseconds = (MR_Integer) ts.tv_sec * 1000000000 + ts.tv_nsec;
").
