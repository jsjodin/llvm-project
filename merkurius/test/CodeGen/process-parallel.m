% A parallel region holds up no worker, and its team is claimed from the
% cores that are free (design document, "OpenMP cooperation"). With one worker,
% a process spawned before the region runs while it does, as the time it
% sends shows, and the team is that one core. A process in a region cannot be
% preempted, so only the move lets it run then. Without -fopenmp there is no
% region, and main is preempted instead. In a program that has not
% spawned, the team is libomp's (omp-parallel-do-run.test).
%
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t
% RUN: %libomp-env MERK_WORKERS=1 OMP_NUM_THREADS=8 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env MERK_WORKERS=4 OMP_NUM_THREADS=8 %t | FileCheck %s --check-prefix=FOUR --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential && env MERK_WORKERS=1 %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module process_parallel.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

:- pragma foreign_decl("C", "
extern int omp_get_num_threads(void) __attribute__((weak));
").

:- func team_size(int) = int.
:- pragma foreign_proc("C",
    team_size(I::in) = (T::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    (void) I;
    T = omp_get_num_threads ? omp_get_num_threads() : 1;
").

:- func work(int, int) = int.
work(N, A) = (A * 3 + N) mod 1000003.

:- pred region(int::in, int::out, int::out) is det.
region(N, Team, Sum) :-
    some [!T, !S] (
        !:T = 0,
        !:S = 0,
        %$omp parallel do reduction(max : T) reduction((+) : S)
        ( for(I, 1, N) do
            !:T = max(!.T, team_size(I)),
            !:S = !.S + work(I, I)
        ),
        Team = !.T,
        Sum = !.S
    ).

:- pred sender(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
sender(Reply, _, !IO) :-
    now(Time, !IO),
    send(Reply, Time, !IO).

:- pred now(int::out, io::di, io::uo) is det.
:- pragma foreign_decl("C", "#include <time.h>").
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    struct timespec s;
    clock_gettime(CLOCK_MONOTONIC, &s);
    T = s.tv_sec * 1000000000 + s.tv_nsec;
").

main(!IO) :-
    new_mailbox(Reply, !IO),
    spawn(sender(Reply), _ : mailbox(int), !IO),
    region(1000000000, Team, Sum),
    now(End, !IO),
    receive(Reply, any, infinity, Got, !IO),
    ( if Sum = 0 then
        io.write_string("no sum\n", !IO)
    else
        ( if Got = message(Time), Time < End then
            io.write_string("ran during the region\n", !IO)
        else
            io.write_string("did not run during the region\n", !IO)
        ),
        ( if Team =< 4 then
            io.write_string("at most four\n", !IO)
        else
            io.write_string("too big\n", !IO)
        ),
        io.write_int(Team, !IO),
        io.nl(!IO)
    ).

% OUTPUT:ran during the region
% OUTPUT-NEXT:at most four
% OUTPUT-NEXT:1
% OUTPUT-NOT:{{.}}

% FOUR:ran during the region
% FOUR-NEXT:at most four
% FOUR-NEXT:{{[1-4]}}
% FOUR-NOT:{{.}}
