% mandel.m's Mandelbrot zoom on several places, GPU processes on one GPU
% with a share of its teams each, that balance a frame's tiles between them
% (docs/INPROGRESS.md, step 4 of the demo, (m3)). The frames, pixels and
% counts are mandel.m's, so the iterations and checksum are too.
%
% - A frame is MANDEL_SIZE pixels square (2048), in tiles of MANDEL_TILE
%   pixels square (128, a power of two that divides it), numbered by rows.
% - MANDEL_PLACES places (4), each with a MANDEL_PLACES'th of the GPU's
%   processes' kernel's teams (process.set_worker_teams). A place steps
%   one tile at a time: a loop across its teams over the tile's pixels,
%   with a reduction of their iterations.
% - The host is the coordinator. It starts each frame, gathers each place's
%   report when the place has no more tiles, and starts the next frame
%   once every place has reported.
%
% MANDEL_POLICY chooses how the tiles are shared:
% - static: each place gets a contiguous range of the frame's tiles, an
%   equal share by rows, as a split in advance would give it, and nothing
%   moves.
% - steal (the default): the static ranges; a place that runs out asks its
%   neighbour round the ring of places. A place with two or more tiles left
%   between two of its tiles answers the thief with the back half of them;
%   one with fewer passes the request on round the ring, and the last
%   before the thief answers none. A place whose request comes back none
%   reports. The ring needs only each place's neighbour's mailbox.
% - pool: the host holds the frame's tiles, and a place that runs out asks
%   it for MANDEL_CHUNK more (4); with none left, the place reports.
%
% The output is the iterations and checksum, the hand-offs of tiles, each
% place's tiles and busy time, by the device's clock, over the zoom, how
% even those times are, and the whole zoom's time, by the host's clock.
% Messages between places go through the host, as GPU processes' messages
% do for now (docs/INPROGRESS.md, GPU processes without omp target, 5d).
%
% Built with -fprocess-declarations, -fdo-loops and --offload-arch, and
% -fopenmp for the loops to run across the teams. The first argument `host`
% spawns the places at the host, for tests. See README.md.

:- module mandel_places.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

    % What a place is told.
:- type order
    --->    frame(int, float, int, int)
            % Frame F, its width, and the first of its tiles for the place
            % and how many.
    ;       next(mailbox(order))
            % The next place round the ring.
    ;       steal(mailbox(order), int)
            % A thief's request, and how many places it has reached.
    ;       tiles(int, int)
            % Tiles for a thief or an asker: the first and how many.
    ;       none
            % No tiles for a thief or an asker.
    ;       stop.

    % What a place tells the host.
:- type report
    --->    ask(int)
            % The place asks for tiles, under the pool.
    ;       idle(int, int, int, float, int)
            % The place has no more tiles: its number, the tiles it stepped
            % in the frame, their iterations, its busy time, and the tiles
            % it was handed.
    ;       wrong(int, int).
            % The place had a message it did not expect, in a state.

:- func static = int.
static = 0.
:- func steal = int.
steal = 1.
:- func pool = int.
pool = 2.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Where = ( if Args = ["host"] then host else gpu(0) ),
    setting("MANDEL_SIZE", 2048, Size, !IO),
    setting("MANDEL_TILE", 128, Tile, !IO),
    setting("MANDEL_FRAMES", 40, Frames, !IO),
    setting("MANDEL_PLACES", 4, K, !IO),
    setting("MANDEL_CHUNK", 4, Chunk, !IO),
    text_setting("MANDEL_POLICY", "steal", PolicyName, !IO),
    Log = log2(Tile, 0),
    ( if
        Tile >= 1, 1 << Log = Tile,
        Size >= Tile, Size mod Tile = 0,
        Frames >= 1,
        K >= 1,
        Chunk >= 1,
        policy(PolicyName, Policy)
    then
        PerRow = Size / Tile,
        header(PolicyName, K, Frames, Size, Tile, !IO),
        set_worker_teams(Where, share(K), !IO),
        new_mailbox(Host, !IO),
        now(Start, !IO),
        list.map_foldl(start_place(Where, Host, K, Size, Log, Policy),
            0 .. K - 1, Places, !IO),
        link(Places, Places, !IO),
        array.init(K, 0, PlaceTiles0),
        array.init(K, 0.0, Busy0),
        zoom(0, Frames, 3.0, Host, Places, K, PerRow * PerRow, Policy, Chunk,
            0, Total, 0, Checksum, 0, Handed, PlaceTiles0, PlaceTiles,
            Busy0, Busy, ok, Outcome, !IO),
        list.foldl(stop_place, Places, !IO),
        now(End, !IO),
        (
            Outcome = ok,
            results(Total, Checksum, Handed, PlaceTiles, Busy, End - Start,
                !IO)
        ;
            Outcome = failed(Message),
            io.write_string(Message, !IO),
            exit_failure(!IO)
        )
    else
        io.write_string("MANDEL_TILE must be a power of two that divides " ++
            "MANDEL_SIZE, MANDEL_FRAMES, MANDEL_PLACES and MANDEL_CHUNK at " ++
            "least 1, and MANDEL_POLICY static, steal or pool.\n", !IO),
        exit_failure(!IO)
    ).

:- pred policy(string::in, int::out) is semidet.
policy("static", static).
policy("steal", steal).
policy("pool", pool).

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

:- pred start_place(place::in, mailbox(report)::in, int::in, int::in, int::in,
    int::in, int::in, mailbox(order)::out, io::di, io::uo) is det.
start_place(Where, Host, K, Size, Log, Policy, P, Box, !IO) :-
    spawn(Where, place(Host, P, K, Size, Log, Policy), Box, !IO).

    % Tells each place its neighbour round the ring.
:- pred link(list(mailbox(order))::in, list(mailbox(order))::in,
    io::di, io::uo) is det.
link([], _, !IO).
link([Box | Boxes], All, !IO) :-
    (
        Boxes = [Next | _]
    ;
        Boxes = [],
        Next = list.det_head(All)
    ),
    send(Box, next(Next), !IO),
    link(Boxes, All, !IO).

:- pred stop_place(mailbox(order)::in, io::di, io::uo) is det.
stop_place(Box, !IO) :-
    send(Box, stop, !IO).

    % Frames F to Frames of width Width on, each started and gathered, with
    % the iterations, the checksum, the tiles handed between places, and
    % each place's tiles and busy time summed.
:- pred zoom(int::in, int::in, float::in, mailbox(report)::in,
    list(mailbox(order))::in, int::in, int::in, int::in, int::in,
    int::in, int::out, int::in, int::out, int::in, int::out,
    array(int)::array_di, array(int)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    outcome::in, outcome::out, io::di, io::uo) is det.
zoom(F, Frames, Width, Host, Places, K, Tiles, Policy, Chunk, !Total,
        !Checksum, !Handed, !PlaceTiles, !Busy, !Outcome, !IO) :-
    ( if F >= Frames ; !.Outcome \= ok then
        true
    else
        start_frame(F, Width, Places, 0, K, Tiles, Policy, !IO),
        Cursor = ( if Policy = pool then 0 else Tiles ),
        gather(K, Host, Places, Tiles, Chunk, Cursor, 0, Stepped, 0, Frame,
            !Handed, !PlaceTiles, !Busy, !Outcome, !IO),
        ( if !.Outcome = ok, Stepped \= Tiles then
            !:Outcome = failed(string.format("Frame %d: the places " ++
                "stepped %d tiles of %d.\n", [i(F), i(Stepped), i(Tiles)]))
        else
            true
        ),
        !:Total = !.Total + Frame,
        !:Checksum = !.Checksum + (F + 1) * Frame,
        zoom(F + 1, Frames, Width * 0.5, Host, Places, K, Tiles, Policy,
            Chunk, !Total, !Checksum, !Handed, !PlaceTiles, !Busy, !Outcome,
            !IO)
    ).

    % Sends each place from the P'th on its tiles of frame F: under the
    % pool none, and otherwise its static range.
:- pred start_frame(int::in, float::in, list(mailbox(order))::in, int::in,
    int::in, int::in, int::in, io::di, io::uo) is det.
start_frame(_, _, [], _, _, _, _, !IO).
start_frame(F, Width, [Box | Boxes], P, K, Tiles, Policy, !IO) :-
    ( if Policy = pool then
        First = 0,
        Count = 0
    else
        First = P * Tiles / K,
        Count = (P + 1) * Tiles / K - First
    ),
    send(Box, frame(F, Width, First, Count), !IO),
    start_frame(F, Width, Boxes, P + 1, K, Tiles, Policy, !IO).

    % Gathers the reports of the Waiting places that have not reported, and
    % under the pool answers their asks from the Cursor'th tile on.
:- pred gather(int::in, mailbox(report)::in, list(mailbox(order))::in,
    int::in, int::in, int::in, int::in, int::out, int::in, int::out,
    int::in, int::out, array(int)::array_di, array(int)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    outcome::in, outcome::out, io::di, io::uo) is det.
gather(Waiting, Host, Places, Tiles, Chunk, Cursor, !Stepped, !Frame,
        !Handed, !PlaceTiles, !Busy, !Outcome, !IO) :-
    ( if Waiting = 0 ; !.Outcome \= ok then
        true
    else
        receive(Host, any, after(600000), R, !IO),
        ( if R = message(idle(P, T, Iterations, Busy, Handed)) then
            !:Stepped = !.Stepped + T,
            !:Frame = !.Frame + Iterations,
            !:Handed = !.Handed + Handed,
            !PlaceTiles ^ elem(P) := !.PlaceTiles ^ elem(P) + T,
            !Busy ^ elem(P) := !.Busy ^ elem(P) + Busy,
            gather(Waiting - 1, Host, Places, Tiles, Chunk, Cursor, !Stepped,
                !Frame, !Handed, !PlaceTiles, !Busy, !Outcome, !IO)
        else if R = message(ask(P)) then
            Box = list.det_index0(Places, P),
            ( if Cursor < Tiles then
                Count = int.min(Chunk, Tiles - Cursor),
                send(Box, tiles(Cursor, Count), !IO),
                Next = Cursor + Count
            else
                send(Box, none, !IO),
                Next = Cursor
            ),
            gather(Waiting, Host, Places, Tiles, Chunk, Next, !Stepped,
                !Frame, !Handed, !PlaceTiles, !Busy, !Outcome, !IO)
        else if R = message(wrong(P, State)) then
            !:Outcome = failed(string.format("Place %d had a message it " ++
                "did not expect, %s.\n", [i(P), s(state_name(State))]))
        else
            !:Outcome = failed("A place did not answer.\n")
        )
    ).

:- func state_name(int) = string.
state_name(S) =
    ( if S = 0 then "before its neighbour"
    else if S = 1 then "between frames"
    else if S = 2 then "while stepping"
    else "while asking" ).

%---------------------------------------------------------------------------%
% The places.

    % Place P of K, for frames of Size pixels square in tiles of 2^Log, under
    % the policy: told its neighbour first, then frames until stop.
:- process place(mailbox(report), int, int, int, int, int)
    receives order on gpu.

place(Host, P, K, Size, Log, Policy, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(next(Next)) then
        between(Host, P, K, Size, Log, Policy, Self, Next, !IO)
    else
        send(Host, wrong(P, 0), !IO)
    ).

    % Between frames: a frame starts, a steal passes on, or stop.
:- pred between(mailbox(report)::in, int::in, int::in, int::in, int::in,
    int::in, mailbox(order)::in, mailbox(order)::in, io::di, io::uo) is det.
between(Host, P, K, Size, Log, Policy, Self, Next, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(frame(F, Width, First, Count)) then
        Step = Width / float(Size),
        work(Host, P, K, Size, Log, Policy, Self, Next, Step, limit(F),
            First, Count, 0, 0, 0.0, 0, !IO)
    else if R = message(steal(Thief, Hops)) then
        pass_on(K, Next, Thief, Hops, !IO),
        between(Host, P, K, Size, Log, Policy, Self, Next, !IO)
    else if R = message(stop) then
        true
    else
        send(Host, wrong(P, 1), !IO)
    ).

    % A steal that this place cannot answer with tiles: on round the ring,
    % or none to the thief from the last place before it.
:- pred pass_on(int::in, mailbox(order)::in, mailbox(order)::in, int::in,
    io::di, io::uo) is det.
pass_on(K, Next, Thief, Hops, !IO) :-
    ( if Hops + 1 < K then
        send(Next, steal(Thief, Hops + 1), !IO)
    else
        send(Thief, none, !IO)
    ).

    % Steps the tiles from First, Count of them, answering a steal between
    % two, then asks for more or reports, with the tiles stepped, their
    % iterations, the busy time and the tiles handed to this place so far
    % in the frame.
:- pred work(mailbox(report)::in, int::in, int::in, int::in, int::in,
    int::in, mailbox(order)::in, mailbox(order)::in, float::in, int::in,
    int::in, int::in, int::in, int::in, float::in, int::in,
    io::di, io::uo) is det.
work(Host, P, K, Size, Log, Policy, Self, Next, Step, Limit, First, Count,
        Tiles, Iterations, Busy, Handed, !IO) :-
    ( if Count > 0 then
        receive(Self, any, after(0), R, !IO),
        ( if R = message(steal(Thief, Hops)) then
            ( if Count >= 2 then
                Give = Count / 2,
                send(Thief, tiles(First + Count - Give, Give), !IO),
                Keep = Count - Give
            else
                pass_on(K, Next, Thief, Hops, !IO),
                Keep = Count
            ),
            Fine = yes
        else if R = timeout then
            Keep = Count,
            Fine = yes
        else
            Keep = Count,
            Fine = no
        ),
        (
            Fine = yes,
            now(T0, !IO),
            tile(Size, Log, Step, Limit, First, Sum),
            now(T1, !IO),
            work(Host, P, K, Size, Log, Policy, Self, Next, Step, Limit,
                First + 1, Keep - 1, Tiles + 1, Iterations + Sum,
                Busy + (T1 - T0), Handed, !IO)
        ;
            Fine = no,
            send(Host, wrong(P, 2), !IO)
        )
    else if Policy = steal, K > 1 then
        send(Next, steal(Self, 1), !IO),
        ask(Host, P, K, Size, Log, Policy, Self, Next, Step, Limit, Tiles,
            Iterations, Busy, Handed, !IO)
    else if Policy = pool then
        send(Host, ask(P), !IO),
        ask(Host, P, K, Size, Log, Policy, Self, Next, Step, Limit, Tiles,
            Iterations, Busy, Handed, !IO)
    else
        report(Host, P, K, Size, Log, Policy, Self, Next, Tiles, Iterations,
            Busy, Handed, !IO)
    ).

:- type fine
    --->    yes
    ;       no.

    % Waits for the answer to a steal or an ask, passing on others' steals.
:- pred ask(mailbox(report)::in, int::in, int::in, int::in, int::in, int::in,
    mailbox(order)::in, mailbox(order)::in, float::in, int::in, int::in,
    int::in, float::in, int::in, io::di, io::uo) is det.
ask(Host, P, K, Size, Log, Policy, Self, Next, Step, Limit, Tiles, Iterations,
        Busy, Handed, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(tiles(First, Count)) then
        work(Host, P, K, Size, Log, Policy, Self, Next, Step, Limit, First,
            Count, Tiles, Iterations, Busy, Handed + Count, !IO)
    else if R = message(none) then
        report(Host, P, K, Size, Log, Policy, Self, Next, Tiles, Iterations,
            Busy, Handed, !IO)
    else if R = message(steal(Thief, Hops)) then
        pass_on(K, Next, Thief, Hops, !IO),
        ask(Host, P, K, Size, Log, Policy, Self, Next, Step, Limit, Tiles,
            Iterations, Busy, Handed, !IO)
    else
        send(Host, wrong(P, 3), !IO)
    ).

:- pred report(mailbox(report)::in, int::in, int::in, int::in, int::in,
    int::in, mailbox(order)::in, mailbox(order)::in, int::in, int::in,
    float::in, int::in, io::di, io::uo) is det.
report(Host, P, K, Size, Log, Policy, Self, Next, Tiles, Iterations, Busy,
        Handed, !IO) :-
    send(Host, idle(P, Tiles, Iterations, Busy, Handed), !IO),
    between(Host, P, K, Size, Log, Policy, Self, Next, !IO).

    % Tile T's iterations, across the place's teams.
    %
:- pred tile(int::in, int::in, float::in, int::in, int::in, int::out) is det.
tile(Size, Log, Step, Limit, T, Total) :-
    Side = 1 << Log,
    PerRow = Size >> Log,
    Top = unchecked_quotient(T, PerRow) << Log,
    Left = unchecked_rem(T, PerRow) << Log,
    Half = Size / 2,
    %$omp distribute parallel do reduction((+) : Total)
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

    % The iterations and checksum; the tiles handed between places; each
    % place's tiles and busy time; the most busy time against the average,
    % 1 when the places are even; and the zoom's time.
:- pred results(int::in, int::in, int::in, array(int)::in, array(float)::in,
    float::in, io::di, io::uo) is det.
results(Total, Checksum, Handed, PlaceTiles, Busy, Time, !IO) :-
    io.format("Iterations: %d\n", [i(Total)], !IO),
    io.format("Checksum: %d\n", [i(Checksum)], !IO),
    io.format("Tiles handed: %d\n", [i(Handed)], !IO),
    K = array.size(Busy),
    ( for(P, 0, K - 1), fromto(!.IO, IO0, IO1, !:IO) do
        place_line(P, PlaceTiles ^ elem(P), Busy ^ elem(P), IO0, IO1)
    ),
    ( for(P, 0, K - 1), fromto(0.0, M0, M1, Most),
        fromto(0.0, S0, S1, Sum) do
        M1 = float.max(M0, Busy ^ elem(P)),
        S1 = S0 + Busy ^ elem(P)
    ),
    Balance = ( if Sum > 0.0 then Most / (Sum / float(K)) else 1.0 ),
    write_results(Balance, Total, Time, !IO).

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

:- pred text_setting(string::in, string::in, string::out, io::di, io::uo)
    is det.
:- pragma foreign_proc("C",
    text_setting(Name::in, Default::in, Value::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    const char *s = getenv(Name);
    Value = s && *s ? (MR_String) s : Default;
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

:- pred header(string::in, int::in, int::in, int::in, int::in, io::di, io::uo)
    is det.
:- pragma foreign_proc("C",
    header(Policy::in, K::in, Frames::in, Size::in, Tile::in,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Mandelbrot zoom, Mercury, %ld places, %s\\n\", (long) K, Policy);
    printf(\"Frames: %ld of %ld x %ld pixels, tiles of %ld x %ld\\n\",
        (long) Frames, (long) Size, (long) Size, (long) Tile, (long) Tile);
").

:- pred place_line(int::in, int::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    place_line(P::in, Tiles::in, Busy::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Place %ld: %ld tiles, busy %.6f s\\n\", (long) P, (long) Tiles,
        Busy);
").

:- pred write_results(float::in, int::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Balance::in, Total::in, Time::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Balance: %.3f\\n\", Balance);
    printf(\"%-12s%-14s\\n\", \"Time (sec)\", \"Giterations/s\");
    printf(\"%-12.6f%-14.3f\\n\", Time, (double) Total / Time * 1.0e-9);
").
