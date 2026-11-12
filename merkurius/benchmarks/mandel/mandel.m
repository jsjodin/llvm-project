% A deep Mandelbrot zoom in one GPU process, dense: each frame a loop
% under distribute parallel do over its pixels across all the process's
% worker teams, with a reduction of their iterations (docs/INPROGRESS.md,
% step 4 of the demo). It is the bar for mandel-places.m's places, which
% split the teams, as mandel.c is for it.
%
% Frame f is MANDEL_SIZE pixels square (2048), of width 3 * 0.5^f, centred
% on a point of the set's boundary, with an escape limit of 256 + 64 * f,
% for MANDEL_FRAMES frames (40). Each pixel's iterations follow one order of
% operations, which merkc does not fuse, so that mandel.c and mandel.py give
% the same counts. MANDEL_TRACE=1 prints each frame's line.
%
% Built with -fprocess-declarations, -fdo-loops and --offload-arch, and
% -fopenmp for the loops to run across the teams; without -fopenmp the
% process runs them on its own lane. The first argument `host` spawns the
% process at the host, for tests. See README.md.

:- module mandel.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    setting("MANDEL_SIZE", 2048, Size, !IO),
    setting("MANDEL_FRAMES", 40, Frames, !IO),
    setting("MANDEL_TRACE", 0, Trace, !IO),
    ( if Size >= 2, Frames >= 1 then
        header(Frames, Size, !IO),
        new_mailbox(TotalsBox, !IO),
        new_mailbox(TimesBox, !IO),
        now(Start, !IO),
        spawn(Place, zoom(TotalsBox, TimesBox, Size, Frames), _, !IO),
        receive(TotalsBox, any, after(600000), R1, !IO),
        receive(TimesBox, any, after(600000), R2, !IO),
        now(End, !IO),
        ( if R1 = message(Totals), R2 = message(Times) then
            ( if Trace = 0 then
                true
            else
                ( for(F, 0, Frames - 1), fromto(!.IO, IO0, IO1, !:IO) do
                    trace_line(F, limit(F), Totals ^ elem(F),
                        Times ^ elem(F), IO0, IO1)
                )
            ),
            ( for(F, 0, Frames - 1), fromto(0, T0, T1, Total),
                fromto(0, C0, C1, Checksum) do
                T1 = T0 + Totals ^ elem(F),
                C1 = C0 + (F + 1) * Totals ^ elem(F)
            ),
            write_results(Total, Checksum, End - Start, !IO)
        else
            io.write_string("The GPU process did not answer.\n", !IO),
            exit_failure(!IO)
        )
    else
        io.write_string("MANDEL_SIZE must be at least 2 and MANDEL_FRAMES " ++
            "at least 1.\n", !IO),
        exit_failure(!IO)
    ).

    % The centre of the zoom, on the set's boundary.
:- func x0 = float.
x0 = -0.743643887037151.
:- func y0 = float.
y0 = 0.131825904205330.

    % Frame F's escape limit.
:- func limit(int) = int.
limit(F) = 256 + 64 * F.

%---------------------------------------------------------------------------%
% The process.

    % Computes the frames and sends the host each one's iterations and its
    % time by the device's clock.
:- process zoom(mailbox(array(int)), mailbox(array(float)), int, int)
    receives int on gpu.

zoom(TotalsBox, TimesBox, Size, Frames, _Self, !IO) :-
    array.init(Frames, 0, Totals0),
    array.init(Frames, 0.0, Times0),
    frames(0, Frames, Size, 3.0, Totals0, Totals, Times0, Times, !IO),
    send(TotalsBox, Totals, !IO),
    send(TimesBox, Times, !IO).

:- pred frames(int::in, int::in, int::in, float::in,
    array(int)::array_di, array(int)::array_uo,
    array(float)::array_di, array(float)::array_uo, io::di, io::uo) is det.
frames(F, Frames, Size, Width, !Totals, !Times, !IO) :-
    ( if F >= Frames then
        true
    else
        now(T0, !IO),
        frame(Size, Width / float(Size), limit(F), Total),
        now(T1, !IO),
        !Totals ^ elem(F) := Total,
        !Times ^ elem(F) := T1 - T0,
        frames(F + 1, Frames, Size, Width * 0.5, !Totals, !Times, !IO)
    ).

    % A frame's iterations: its pixels' sum, across the teams.
    %
:- pred frame(int::in, float::in, int::in, int::out) is det.
frame(Size, Step, Limit, Total) :-
    Half = Size / 2,
    %$omp distribute parallel do reduction((+) : Total)
    ( for(K, 0, Size * Size - 1), fromto(0, S0, S1, Total) do
        PY = unchecked_quotient(K, Size),
        PX = unchecked_rem(K, Size),
        CX = x0 + float(PX - Half) * Step,
        CY = y0 + float(PY - Half) * Step,
        S1 = S0 + escape(CX, CY, Limit, 0, 0.0, 0.0)
    ).

    % The iterations before z, at X and Y after N of them, escapes from c,
    % at most Limit.
    %
:- func escape(float, float, int, int, float, float) = int.
escape(CX, CY, Limit, N, X, Y) = Iterations :-
    ( if N >= Limit then
        Iterations = Limit
    else
        X2 = X * X,
        Y2 = Y * Y,
        ( if X2 + Y2 > 4.0 then
            Iterations = N
        else
            Iterations = escape(CX, CY, Limit, N + 1, X2 - Y2 + CX,
                (2.0 * X) * Y + CY)
        )
    ).

%---------------------------------------------------------------------------%
% What C does for the program: the clock, the environment and printf. The
% clock is libc's, on the device as on the host.

:- pragma foreign_decl("C", "#include <stdio.h>").
:- pragma foreign_decl("C", "#include <stdlib.h>").
:- pragma foreign_decl("C", "#include <time.h>").

:- pred now(float::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    T = (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
").

% The environment variable's value, if it is set, and otherwise Default.
:- pred setting(string::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    setting(Name::in, Default::in, Value::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    const char *s = getenv(Name);
    Value = s && *s ? (MR_Integer) strtol(s, NULL, 10) : Default;
").

% Exits with status 1, since io.set_exit_status is not implemented yet.
:- pred exit_failure(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    exit_failure(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    fflush(stdout);
    exit(1);
").

:- pred header(int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    header(Frames::in, Size::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Mandelbrot zoom, Mercury, GPU process\\n\");
    printf(\"Frames: %ld of %ld x %ld pixels\\n\", (long) Frames, (long) Size,
        (long) Size);
").

:- pred trace_line(int::in, int::in, int::in, float::in, io::di, io::uo)
    is det.
:- pragma foreign_proc("C",
    trace_line(F::in, Limit::in, Iterations::in, Time::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"frame %ld: limit %ld, %ld iterations, %.6f s\\n\", (long) F,
        (long) Limit, (long) Iterations, Time);
").

    % The time is the host's, from before the spawn to the last answer, as
    % mandel.c's covers its frames' kernels and their launches.
:- pred write_results(int::in, int::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Total::in, Checksum::in, Time::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Iterations: %ld\\n\", (long) Total);
    printf(\"Checksum: %ld\\n\", (long) Checksum);
    printf(\"%-12s%-14s\\n\", \"Time (sec)\", \"Giterations/s\");
    printf(\"%-12.6f%-14.3f\\n\", Time, (double) Total / Time * 1.0e-9);
").
