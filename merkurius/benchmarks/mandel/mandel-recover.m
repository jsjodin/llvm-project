% mandel-places.m's zoom under its pool policy, with places that may end at
% any time, and a coordinator that recovers their tiles (docs/INPROGRESS.md,
% Failure, step 4, D1). The frames, pixels and counts are mandel.m's, so the
% iterations and checksum are too, whichever places end and whenever.
%
% - A frame is MANDEL_SIZE pixels square (2048), in tiles of MANDEL_TILE
%   pixels square (128), numbered by rows, on MANDEL_PLACES places (4), each
%   with a MANDEL_PLACES'th of the GPU's processes' kernel's teams.
% - The host, the coordinator, holds each frame's tiles, and gives a place
%   that asks MANDEL_CHUNK of them (4). A place steps its chunk, one tile at
%   a time, and reports the chunk done with its iterations, which is its
%   next ask.
% - The host monitors its places and keeps the chunk each was given. When
%   one ends other than normally, its chunk goes back to the pool, ahead of
%   the tiles not yet given, and a new place takes its share of the teams
%   and its number, up to MANDEL_RESTARTS times (1000). A chunk counts once,
%   from the place it was given to: what a place that has ended reported
%   after its end is dropped, as its incarnation is no longer the place's.
%
% Places end through process.stop, by MERK_FAULTS (fail=order, or kernel on
% a GPU), or any other way. A GPU place reads its stop word in its receives,
% so it ends between chunks, never in the middle of one.
%
% The output is the iterations and checksum, the places lost and the tiles
% given back, and the whole zoom's time, by the host's clock. mandel-places
% under MANDEL_POLICY=pool, with no faults, is the run to measure against
% (compare-recover.sh).
%
% Built with -fprocess-declarations, -fdo-loops and --offload-arch, and
% -fopenmp for the loops to run across the teams. The first argument `host`
% spawns the places at the host, for tests. See README.md.

:- module mandel_recover.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list, process, string.

    % What a place is told.
:- type order
    --->    frame(int, float)
            % Frame F starts, of this width: ask for tiles.
    ;       tiles(int, int)
            % The first of the tiles to step and how many.
    ;       stop.

    % What a place tells the host: its number and its incarnation, which
    % the host gave it, and an ask at frame F's start, or a chunk done, with
    % the chunk's first tile, its count, iterations and the busy time. A
    % place that starts in the middle of a frame may ask for it only after
    % its tiles are done, so an ask names its frame.
:- type report
    --->    ask(int, int, int)
    ;       done(int, int, int, int, int, float).

    % What the host reads: the places' reports, and their ends, from
    % another mailbox, as a down's text has no fixed size, which a GPU
    % process's messages must.
:- type happening
    --->    report(report)
    ;       lost(int, int, down_info).

:- func as_report(report) = happening.
as_report(R) = report(R).

:- func same(happening) = happening.
same(H) = H.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Where = ( if Args = ["host"] then host else gpu(0) ),
    setting("MANDEL_SIZE", 2048, Size, !IO),
    setting("MANDEL_TILE", 128, Tile, !IO),
    setting("MANDEL_FRAMES", 40, Frames, !IO),
    setting("MANDEL_PLACES", 4, K, !IO),
    setting("MANDEL_CHUNK", 4, Chunk, !IO),
    setting("MANDEL_RESTARTS", 1000, Restarts, !IO),
    Log = log2(Tile, 0),
    ( if
        Tile >= 1, 1 << Log = Tile,
        Size >= Tile, Size mod Tile = 0,
        Frames >= 1,
        K >= 1,
        Chunk >= 1,
        Restarts >= 0
    then
        PerRow = Size / Tile,
        header(K, Frames, Size, Tile, !IO),
        set_worker_teams(Where, share(K), !IO),
        new_mailbox(Reports, !IO),
        new_mailbox(Downs, !IO),
        Setup = setup(Where, Reports, Downs, Size, Log, PerRow * PerRow,
            Chunk),
        now(Start, !IO),
        list.map_foldl(start_place(Setup), 0 .. K - 1, Slots0, !IO),
        Run0 = run(Slots0, K, Restarts, 0, 0, 0, 0),
        zoom(Setup, 0, Frames, 3.0, Run0, Run, ok, Outcome, !IO),
        Run = run(Slots, _, _, Lost, Back, Total, Checksum),
        list.foldl(stop_place, Slots, !IO),
        now(End, !IO),
        close_mailbox(Downs, !IO),
        close_mailbox(Reports, !IO),
        (
            Outcome = ok,
            results(Total, Checksum, Lost, Back, End - Start, !IO)
        ;
            Outcome = failed(Message),
            io.write_string(Message, !IO),
            exit_failure(!IO)
        )
    else
        io.write_string("MANDEL_TILE must be a power of two that divides " ++
            "MANDEL_SIZE, MANDEL_FRAMES, MANDEL_PLACES and MANDEL_CHUNK at " ++
            "least 1, and MANDEL_RESTARTS at least 0.\n", !IO),
        exit_failure(!IO)
    ).

    % The log of N to base 2, if N is a power of two.
:- func log2(int, int) = int.
log2(N, L) = ( if 1 << L >= N then L else log2(N, L + 1) ).

    % The centre of the zoom, on the set's boundary, as in mandel.m.
:- func x0 = float.
x0 = -0.743643887037151.
:- func y0 = float.
y0 = 0.131825904205330.

    % Frame F's escape limit.
:- func limit(int) = int.
limit(F) = 256 + 64 * F.

%---------------------------------------------------------------------------%
% The host, the coordinator.

:- type outcome
    --->    ok
    ;       failed(string).

    % What stays the same through the zoom: where places go, the host's
    % mailboxes, the frame's size, its tiles' log, its tiles, and a chunk.
:- type setup
    --->    setup(place, mailbox(report), mailbox(happening), int, int, int,
                int).

    % A chunk of tiles: the first and how many; none when 0.
:- type chunk
    --->    chunk(int, int).

    % A place: its mailbox, its incarnation, and the chunk it was given and
    % has not reported.
:- type slot
    --->    slot(mailbox(order), int, chunk).

    % Across the zoom: the places, by number; the next incarnation; the
    % restarts left; the places lost and the tiles given back; and the
    % iterations and checksum.
:- type run
    --->    run(list(slot), int, int, int, int, int, int).

    % Within a frame: the places waiting for tiles, the chunks given back,
    % the next tile not yet given, and the tiles stepped and their
    % iterations.
:- type frame
    --->    frame_state(list(int), list(chunk), int, int, int).

:- pred start_place(setup::in, int::in, slot::out, io::di, io::uo) is det.
start_place(Setup, P, Slot, !IO) :-
    spawn_place(Setup, P, P, Slot, !IO).

    % Place P as incarnation Id, monitored.
:- pred spawn_place(setup::in, int::in, int::in, slot::out, io::di, io::uo)
    is det.
spawn_place(Setup, P, Id, slot(Box, Id, chunk(0, 0)), !IO) :-
    Setup = setup(Where, Reports, Downs, Size, Log, _, _),
    spawn(Where, place(Reports, P, Id, Size, Log), Box, !IO),
    monitor(Box, Downs, (func(D) = lost(P, Id, D)), _, !IO).

:- pred stop_place(slot::in, io::di, io::uo) is det.
stop_place(slot(Box, _, _), !IO) :-
    send(Box, stop, !IO).

    % Frames F to Frames of width Width on, each started and stepped.
:- pred zoom(setup::in, int::in, int::in, float::in, run::in, run::out,
    outcome::in, outcome::out, io::di, io::uo) is det.
zoom(Setup, F, Frames, Width, !Run, !Outcome, !IO) :-
    ( if F >= Frames ; !.Outcome \= ok then
        true
    else
        !.Run = run(Slots, _, _, _, _, _, _),
        list.foldl(start_frame(F, Width), Slots, !IO),
        Setup = setup(_, _, _, _, _, Tiles, _),
        step(Setup, F, Width, frame_state([], [], 0, 0, 0), Frame, !Run,
            !Outcome, !IO),
        Frame = frame_state(_, _, _, Stepped, Iterations),
        ( if !.Outcome = ok, Stepped \= Tiles then
            !:Outcome = failed(string.format("Frame %d: the places " ++
                "stepped %d tiles of %d.\n", [i(F), i(Stepped), i(Tiles)]))
        else
            true
        ),
        !.Run = run(S, N, R, L, B, Total, Checksum),
        !:Run = run(S, N, R, L, B, Total + Iterations,
            Checksum + (F + 1) * Iterations),
        zoom(Setup, F + 1, Frames, Width * 0.5, !Run, !Outcome, !IO)
    ).

:- pred start_frame(int::in, float::in, slot::in, io::di, io::uo) is det.
start_frame(F, Width, slot(Box, _, _), !IO) :-
    send(Box, frame(F, Width), !IO).

    % Answers the places' asks and reports, and their ends, until the
    % frame's tiles are stepped.
:- pred step(setup::in, int::in, float::in, frame::in, frame::out,
    run::in, run::out, outcome::in, outcome::out, io::di, io::uo) is det.
step(Setup, F, Width, !Frame, !Run, !Outcome, !IO) :-
    Setup = setup(_, Reports, Downs, _, _, Tiles, _),
    !.Frame = frame_state(_, _, _, Stepped, _),
    ( if Stepped >= Tiles ; !.Outcome \= ok then
        true
    else
        Selector = selecting(Reports, as_report,
            selecting(Downs, same, new_selector)),
        receive(Selector, after(600000), Got, !IO),
        ( if Got = message(report(ask(P, Id, AskF))) then
            ( if AskF = F, current(!.Run, P, Id) then
                give(Setup, P, !Frame, !Run, !IO)
            else
                true
            )
        else if Got = message(report(done(P, Id, First, Count, Sum, _))) then
            ( if
                current(!.Run, P, Id),
                outstanding(!.Run, P) = chunk(First, Count)
            then
                !.Frame = frame_state(W, Back, Cursor, S0, I0),
                !:Frame = frame_state(W, Back, Cursor, S0 + Count, I0 + Sum),
                set_outstanding(P, chunk(0, 0), !Run),
                give(Setup, P, !Frame, !Run, !IO)
            else
                true
            )
        else if Got = message(lost(P, Id, Down)) then
            ( if current(!.Run, P, Id) then
                lose(Setup, F, Width, P, Down, !Frame, !Run, !Outcome, !IO)
            else
                true
            )
        else
            !:Outcome = failed("No place reported in 10 minutes.\n")
        ),
        step(Setup, F, Width, !Frame, !Run, !Outcome, !IO)
    ).

    % Gives place P a chunk, from those given back first, or puts it on the
    % list of those waiting if no tiles are left to give.
:- pred give(setup::in, int::in, frame::in, frame::out, run::in, run::out,
    io::di, io::uo) is det.
give(Setup, P, !Frame, !Run, !IO) :-
    Setup = setup(_, _, _, _, _, Tiles, Chunk),
    !.Frame = frame_state(Waiting, Back, Cursor, S, I),
    (
        Back = [Given | Rest],
        !:Frame = frame_state(Waiting, Rest, Cursor, S, I),
        hand(P, Given, !Run, !IO)
    ;
        Back = [],
        ( if Cursor < Tiles then
            Count = int.min(Chunk, Tiles - Cursor),
            !:Frame = frame_state(Waiting, Back, Cursor + Count, S, I),
            hand(P, chunk(Cursor, Count), !Run, !IO)
        else
            !:Frame = frame_state([P | Waiting], Back, Cursor, S, I)
        )
    ).

:- pred hand(int::in, chunk::in, run::in, run::out, io::di, io::uo) is det.
hand(P, Chunk, !Run, !IO) :-
    Chunk = chunk(First, Count),
    !.Run = run(Slots, _, _, _, _, _, _),
    list.det_index0(Slots, P) = slot(Box, _, _),
    send(Box, tiles(First, Count), !IO),
    set_outstanding(P, Chunk, !Run).

    % Place P has ended: its chunk goes back, and to a waiting place if
    % there is one; a new place takes its number and starts the frame.
:- pred lose(setup::in, int::in, float::in, int::in, down_info::in,
    frame::in, frame::out, run::in, run::out, outcome::in, outcome::out,
    io::di, io::uo) is det.
lose(Setup, F, Width, P, Down, !Frame, !Run, !Outcome, !IO) :-
    chunk(First, Count) = outstanding(!.Run, P),
    !.Run = run(Slots, Next, Restarts, Lost, Back0, Total, Checksum),
    !.Frame = frame_state(Waiting0, Back, Cursor, S, I),
    list.delete_all(Waiting0, P, Waiting1),
    ( if Count > 0 then
        Back1 = [chunk(First, Count) | Back],
        Back2 = Back0 + Count
    else
        Back1 = Back,
        Back2 = Back0
    ),
    !:Frame = frame_state(Waiting1, Back1, Cursor, S, I),
    ( if Restarts > 0 then
        spawn_place(Setup, P, Next, Slot, !IO),
        Slot = slot(Box, _, _),
        list.det_replace_nth(Slots, P + 1, Slot, Slots1),
        !:Run = run(Slots1, Next + 1, Restarts - 1, Lost + 1, Back2, Total,
            Checksum),
        send(Box, frame(F, Width), !IO),
        rehand(Setup, !Frame, !Run, !IO)
    else
        !:Outcome = failed(string.format("Place %d ended, %s, with no " ++
            "restarts left.\n", [i(P), s(reason_text(reason(Down)))]))
    ).

    % Gives the chunks given back to the places waiting, while both last.
:- pred rehand(setup::in, frame::in, frame::out, run::in, run::out,
    io::di, io::uo) is det.
rehand(Setup, !Frame, !Run, !IO) :-
    !.Frame = frame_state(Waiting, Back, Cursor, S, I),
    ( if Waiting = [P | Rest], Back = [_ | _] then
        !:Frame = frame_state(Rest, Back, Cursor, S, I),
        give(Setup, P, !Frame, !Run, !IO),
        rehand(Setup, !Frame, !Run, !IO)
    else
        true
    ).

    % Whether place P's incarnation is Id.
:- pred current(run::in, int::in, int::in) is semidet.
current(run(Slots, _, _, _, _, _, _), P, Id) :-
    list.index0(Slots, P, slot(_, Id, _)).

:- func outstanding(run, int) = chunk.
outstanding(run(Slots, _, _, _, _, _, _), P) = Chunk :-
    list.det_index0(Slots, P) = slot(_, _, Chunk).

:- pred set_outstanding(int::in, chunk::in, run::in, run::out) is det.
set_outstanding(P, Chunk, !Run) :-
    !.Run = run(Slots, Next, Restarts, Lost, Back, Total, Checksum),
    list.det_index0(Slots, P) = slot(Box, Id, _),
    list.det_replace_nth(Slots, P + 1, slot(Box, Id, Chunk), Slots1),
    !:Run = run(Slots1, Next, Restarts, Lost, Back, Total, Checksum).

%---------------------------------------------------------------------------%
% The places.

    % Place P, incarnation Id, for frames of Size pixels square in tiles of
    % 2^Log: asks for tiles at each frame's start, steps each chunk it is
    % given and reports it, until stop.
:- process place(mailbox(report), int, int, int, int) receives order on gpu.

place(Host, P, Id, Size, Log, Self, !IO) :-
    wait(Host, P, Id, Size, Log, Self, 0.0, 0, !IO).

    % Waits for an order, with the frame's step and limit.
:- pred wait(mailbox(report)::in, int::in, int::in, int::in, int::in,
    mailbox(order)::in, float::in, int::in, io::di, io::uo) is det.
wait(Host, P, Id, Size, Log, Self, Step, Limit, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(frame(F, Width)) then
        send(Host, ask(P, Id, F), !IO),
        wait(Host, P, Id, Size, Log, Self, Width / float(Size), limit(F),
            !IO)
    else if R = message(tiles(First, Count)) then
        now(T0, !IO),
        chunk(Size, Log, Step, Limit, First, First + Count, 0, Sum),
        now(T1, !IO),
        send(Host, done(P, Id, First, Count, Sum, T1 - T0), !IO),
        wait(Host, P, Id, Size, Log, Self, Step, Limit, !IO)
    else
        true
    ).

    % The iterations of tiles T to End, before End.
:- pred chunk(int::in, int::in, float::in, int::in, int::in, int::in,
    int::in, int::out) is det.
chunk(Size, Log, Step, Limit, T, End, !Sum) :-
    ( if T >= End then
        true
    else
        tile(Size, Log, Step, Limit, T, S),
        !:Sum = !.Sum + S,
        chunk(Size, Log, Step, Limit, T + 1, End, !Sum)
    ).

    % Tile T's iterations, across the place's teams.
    %
:- pred tile(int::in, int::in, float::in, int::in, int::in, int::out) is det.
tile(Size, Log, Step, Limit, T, Total) :-
    Side = 1 << Log,
    PerRow = Size >> Log,
    Top = unchecked_quotient(T, PerRow) << Log,
    Left = unchecked_rem(T, PerRow) << Log,
    Half = Size / 2,
    %$omp teams distribute parallel do reduction((+) : Total)
    ( for(K, 0, Side * Side - 1), fromto(0, S0, S1, Total) do
        PY = Top + (K >> Log),
        PX = Left + (K /\ (Side - 1)),
        CX = x0 + float(PX - Half) * Step,
        CY = y0 + float(PY - Half) * Step,
        S1 = S0 + escape(CX, CY, Limit, 0, 0.0, 0.0)
    ).

    % The iterations before z, at X and Y after N of them, escapes from c,
    % at most Limit, as in mandel.m.
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
% The host's output.

:- pred results(int::in, int::in, int::in, int::in, float::in,
    io::di, io::uo) is det.
results(Total, Checksum, Lost, Back, Time, !IO) :-
    io.format("Iterations: %d\n", [i(Total)], !IO),
    io.format("Checksum: %d\n", [i(Checksum)], !IO),
    io.format("Places lost: %d\n", [i(Lost)], !IO),
    io.format("Tiles given back: %d\n", [i(Back)], !IO),
    write_results(Total, Time, !IO).

%---------------------------------------------------------------------------%
% What C does for the program: the clock, the environment and printf. The
% clock is libc's, on the device as on the host.

:- pragma foreign_decl("C", "#include <stdio.h>").
:- pragma foreign_decl("C", "#include <stdlib.h>").
:- pragma foreign_decl("C", "#include <time.h>").

:- pred now(float::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    T = (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
").

% The environment variable's value, if it is set, and otherwise Default.
:- pred setting(string::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    setting(Name::in, Default::in, Value::out, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    const char *s = getenv(Name);
    Value = s && *s ? (MR_Integer) strtol(s, NULL, 10) : Default;
").

:- pred text_setting(string::in, string::in, string::out, io::di, io::uo)
    is det.
:- pragma foreign_proc("C",
    text_setting(Name::in, Default::in, Value::out, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    const char *s = getenv(Name);
    Value = s && *s ? (MR_String) s : Default;
").

% Exits with status 1, since io.set_exit_status is not implemented yet.
:- pred exit_failure(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    exit_failure(_IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    fflush(stdout);
    exit(1);
").

:- pred header(int::in, int::in, int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    header(K::in, Frames::in, Size::in, Tile::in, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    printf(\"Mandelbrot zoom, Merkurius, %ld places, recovering\\n\", (long) K);
    printf(\"Frames: %ld of %ld x %ld pixels, tiles of %ld x %ld\\n\",
        (long) Frames, (long) Size, (long) Size, (long) Tile, (long) Tile);
").

:- pred write_results(int::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Total::in, Time::in, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    printf(\"%-12s%-14s\\n\", \"Time (sec)\", \"Giterations/s\");
    printf(\"%-12.6f%-14.3f\\n\", Time, (double) Total / Time * 1.0e-9);
").
