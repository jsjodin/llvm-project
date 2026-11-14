% A pipeline of three GPU processes, with feedback: what a frame must cost a
% stage for passing it on to pay (docs/INPROGRESS.md, A pipeline of
% processes). pipeline.hip is the same stages as kernels in one stream.
%
% - The source makes each frame of PIPE_WORDS floats and does the first
%   stage's work on it; the middle process does the second's; the sink the
%   third's, and the frame's checksum. Each passes the frame on in a
%   message, frame(F, Cells).
% - A stage's work on a cell is PIPE_WORK rounds of x = x * a + b, its own
%   a and b, across the process's teams.
% - With PIPE_FEEDBACK=D above 0, the sink sends each frame's checksum back
%   to the source, which waits for frame F - D's before it makes frame F,
%   and mixes it into the frame's cells: D frames are in flight at most.
% - PIPE_MODE=single runs the three stages one after another in one
%   process, the frames in turn, with no messages: the same cells.
% - The checksum is the sum, modulo 2^64, over the frames and their cells
%   of bits(x) (2 i + 1), i the cell's index, which no order changes.
%
% The sink, or the single process, reads the device's clock after each
% frame; the time a frame is the average from the first frame's end to the
% last's, which leaves out the processes' start.
%
% Built with -fprocess-declarations, -fdo-loops and --offload-arch, and
% -fopenmp for the loops to run across the teams. The first argument
% `host` spawns the processes at the host, for tests. See README.md.

:- module pipeline.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

    % What the source receives: the middle process's mailbox, first; then,
    % with feedback, frame F's checksum.
:- type to_source
    --->    next(mailbox(frame))
    ;       seen(int, int).

    % Frame F's cells.
:- type frame
    --->    frame(int, array(float)).

    % The checksum, and the device's clock after the first frame and after
    % the last.
:- type outcome
    --->    outcome(int, float, float).

    % A pipeline's settings, which the processes take as integers, as a
    % GPU process kind takes no records yet: the cells of a frame, the rounds of a stage's
    % work on a cell, the frames, and the frames in flight with feedback, 0
    % for none.
:- type settings
    --->    settings(
                words :: int,
                work :: int,
                frames :: int,
                feedback :: int
            ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Where = ( if Args = ["host"] then host else gpu(0) ),
    setting("PIPE_WORDS", 65536, Words, !IO),
    setting("PIPE_WORK", 64, Work, !IO),
    setting("PIPE_FRAMES", 200, Frames, !IO),
    setting("PIPE_FEEDBACK", 0, Feedback, !IO),
    get_mode(Single, !IO),
    ( if
        Words >= 1, Work >= 0, Frames >= 2, Feedback >= 0, Single >= 0
    then
        header(Words, Work, Frames, Feedback, Single, !IO),
        new_mailbox(Me, !IO),
        ( if Single = 1 then
            set_worker_teams(Where, share(1), !IO),
            spawn(Where, single(Me, Words, Work, Frames, Feedback), _, !IO)
        else
            set_worker_teams(Where, share(3), !IO),
            spawn(Where, source(Words, Work, Frames, Feedback), Source,
                !IO),
            spawn(Where, sink(Me, Source, Words, Work, Frames, Feedback),
                Sink, !IO),
            spawn(Where, middle(Sink, Words, Work, Frames, Feedback),
                Middle, !IO),
            send(Source, next(Middle), !IO)
        ),
        receive(Me, any, after(600000), R, !IO),
        ( if R = message(outcome(Checksum, First, Last)) then
            results(Frames, Checksum, First, Last, !IO)
        else
            io.write_string("The pipeline did not answer.\n", !IO),
            io.set_exit_status(1, !IO)
        )
    else
        io.write_string("PIPE_WORDS must be at least 1, PIPE_WORK and " ++
            "PIPE_FEEDBACK at least 0, PIPE_FRAMES at least 2, and " ++
            "PIPE_MODE pipeline or single.\n", !IO),
        io.set_exit_status(1, !IO)
    ).

%---------------------------------------------------------------------------%
% The processes.

    % Makes each frame and does the first stage's work on it.
:- process source(int, int, int, int) receives to_source on gpu.

source(Words, Work, Frames, Feedback, Self, !IO) :-
    S = settings(Words, Work, Frames, Feedback),
    receive(Self, any, infinity, R, !IO),
    ( if R = message(next(Next)) then
        produce(S, Self, Next, 0, !IO)
    else
        true
    ).

:- pred produce(settings::in, mailbox(to_source)::in, mailbox(frame)::in,
    int::in, io::di, io::uo) is det.
produce(S, Self, Next, F, !IO) :-
    ( if F >= S ^ frames then
        true
    else
        D = S ^ feedback,
        ( if D > 0, F >= D then
            receive(Self, any, infinity, R, !IO),
            Mix = ( if R = message(seen(_, C)) then C /\ 1023 else 0 )
        else
            Mix = 0
        ),
        array.init(S ^ words, 0.0, Cells0),
        make(S, F, Mix, Cells0, Cells),
        send(Next, frame(F, Cells), !IO),
        produce(S, Self, Next, F + 1, !IO)
    ).

    % The second stage.
:- process middle(mailbox(frame), int, int, int, int) receives frame on gpu.

middle(Next, Words, Work, Frames, Feedback, Self, !IO) :-
    S = settings(Words, Work, Frames, Feedback),
    pass(Next, S, Self, 0, !IO).

:- pred pass(mailbox(frame)::in, settings::in, mailbox(frame)::in, int::in,
    io::di, io::uo) is det.
pass(Next, S, Self, K, !IO) :-
    ( if K >= S ^ frames then
        true
    else
        receive(Self, any, infinity, R, !IO),
        ( if R = message(frame(F, In)) then
            array.init(S ^ words, 0.0, Out0),
            stage(S, 1.0000001, -0.0000003, In, Out0, Out),
            send(Next, frame(F, Out), !IO)
        else
            true
        ),
        pass(Next, S, Self, K + 1, !IO)
    ).

    % The third stage and the checksum, sent back to the source with
    % feedback, and the outcome to Main at the end.
:- process sink(mailbox(outcome), mailbox(to_source), int, int, int, int)
    receives frame on gpu.

sink(Main, Source, Words, Work, Frames, Feedback, Self, !IO) :-
    S = settings(Words, Work, Frames, Feedback),
    drain(Source, S, Self, 0, 0, Checksum, 0.0, First, !IO),
    now(Last, !IO),
    send(Main, outcome(Checksum, First, Last), !IO).

:- pred drain(mailbox(to_source)::in, settings::in, mailbox(frame)::in,
    int::in, int::in, int::out, float::in, float::out, io::di, io::uo)
    is det.
drain(Source, S, Self, K, Sum0, Sum, First0, First, !IO) :-
    ( if K >= S ^ frames then
        Sum = Sum0,
        First = First0
    else
        receive(Self, any, infinity, R, !IO),
        ( if R = message(frame(F, In)) then
            array.init(S ^ words, 0.0, Out0),
            stage(S, 0.9999997, 0.0000002, In, Out0, Out),
            C = checksum(S ^ words, Out),
            ( if S ^ feedback > 0 then
                send(Source, seen(F, C), !IO)
            else
                true
            )
        else
            C = 0
        ),
        ( if K = 0 then
            now(First1, !IO)
        else
            First1 = First0
        ),
        drain(Source, S, Self, K + 1, Sum0 + C, Sum, First1, First, !IO)
    ).

    % The three stages, one after another in one process, each frame in
    % turn.
:- process single(mailbox(outcome), int, int, int, int) receives int on gpu.

single(Main, Words, Work, Frames, Feedback, _Self, !IO) :-
    S = settings(Words, Work, Frames, Feedback),
    array.init(int.max(S ^ feedback, 1), 0, Ring),
    whole(S, 0, 0, 0.0, Sum, First, Ring, _, !IO),
    now(Last, !IO),
    send(Main, outcome(Sum, First, Last), !IO).

    % From frame F on, the checksums of the last D frames in Ring, for the
    % feedback that the source mixes in.
:- pred whole(settings::in, int::in, int::in, float::in, int::out,
    float::out, array(int)::array_di, array(int)::array_uo,
    io::di, io::uo) is det.
whole(S, F, Sum0, First0, Sum, First, !Ring, !IO) :-
    ( if F >= S ^ frames then
        Sum = Sum0,
        First = First0
    else
        D = S ^ feedback,
        Mix = ( if D > 0, F >= D then
            !.Ring ^ elem(F mod D) /\ 1023 else 0 ),
        array.init(S ^ words, 0.0, A0),
        make(S, F, Mix, A0, A),
        array.init(S ^ words, 0.0, B0),
        stage(S, 1.0000001, -0.0000003, A, B0, B),
        array.init(S ^ words, 0.0, C0),
        stage(S, 0.9999997, 0.0000002, B, C0, C),
        Check = checksum(S ^ words, C),
        ( if D > 0 then
            !Ring ^ elem(F mod D) := Check
        else
            true
        ),
        ( if F = 0 then
            now(First1, !IO)
        else
            First1 = First0
        ),
        whole(S, F + 1, Sum0 + Check, First1, Sum, First, !Ring, !IO)
    ).

%---------------------------------------------------------------------------%
% The stages' work, across the process's teams.

    % Frame F's cells, Mix mixed in, after the first stage's work.
    %
:- pred make(settings::in, int::in, int::in,
    array(float)::array_di, array(float)::array_uo) is det.
make(S, F, Mix, !Cells) :-
    N = S ^ words,
    Work = S ^ work,
    %$omp distribute parallel do
    ( for(I, 0, N - 1) do
        X0 = float((F * N + I + Mix) /\ 1023) * 0.001,
        !Cells ^ unsafe_elem(I) := rounds(Work, 0.9999999, 0.0000001, X0)
    ).

    % Out's cells from In's, each after Work rounds of x * A + B.
    %
:- pred stage(settings::in, float::in, float::in, array(float)::in,
    array(float)::array_di, array(float)::array_uo) is det.
stage(S, A, B, In, !Out) :-
    N = S ^ words,
    Work = S ^ work,
    %$omp distribute parallel do
    ( for(I, 0, N - 1) do
        !Out ^ unsafe_elem(I) :=
            rounds(Work, A, B, In ^ unsafe_elem(I))
    ).

:- func rounds(int, float, float, float) = float.
rounds(Work, A, B, X0) = X :-
    ( for(_, 1, Work), fromto(X0, Y0, Y1, X) do
        Y1 = Y0 * A + B
    ).

    % The frame's checksum, as the top of the module says.
    %
:- func checksum(int, array(float)) = int.
:- mode checksum(in, in) = out is det.
checksum(N, Cells) = Sum :-
    %$omp distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0, S0, S1, Sum) do
        S1 = S0 + bits(Cells ^ unsafe_elem(I)) * (2 * I + 1)
    ).

%---------------------------------------------------------------------------%
% The host's output.

:- pred results(int::in, int::in, float::in, float::in, io::di, io::uo)
    is det.
results(Frames, Checksum, First, Last, !IO) :-
    write_results(Checksum, (Last - First) / float(Frames - 1), !IO).

%---------------------------------------------------------------------------%
% What C does for the program.

:- pragma foreign_decl("C", "#include <stdio.h>").
:- pragma foreign_decl("C", "#include <stdlib.h>").
:- pragma foreign_decl("C", "#include <string.h>").
:- pragma foreign_decl("C", "#include <time.h>").

    % The float's bits, -0.0 as 0.0.
    %
:- func bits(float) = int.
:- pragma foreign_proc("C",
    bits(F::in) = (B::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    double y = F + 0.0;
    memcpy(&B, &y, sizeof y);
").

:- pred now(float::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
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

    % PIPE_MODE: 0 for pipeline, the default, 1 for single, -1 otherwise.
:- pred get_mode(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    get_mode(Mode::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    const char *s = getenv(\"PIPE_MODE\");
    Mode = !s || !*s || strcmp(s, \"pipeline\") == 0 ? 0
        : strcmp(s, \"single\") == 0 ? 1 : -1;
").

:- pred header(int::in, int::in, int::in, int::in, int::in, io::di, io::uo)
    is det.
:- pragma foreign_proc("C",
    header(Words::in, Work::in, Frames::in, Feedback::in, Single::in,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Pipeline, Mercury, %s\\n\", Single ? \"one GPU process\" :
        \"three GPU processes\");
    printf(\"Frames: %ld of %ld words, %ld rounds a stage\\n\",
        (long) Frames, (long) Words, (long) Work);
    if (Feedback > 0)
        printf(\"Feedback: %ld frames in flight\\n\", (long) Feedback);
    else
        printf(\"Feedback: none\\n\");
").

:- pred write_results(int::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Checksum::in, Frame::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Checksum: %016llx\\n\", (unsigned long long) Checksum);
    printf(\"Time a frame: %.1f us\\n\", Frame * 1.0e6);
").
