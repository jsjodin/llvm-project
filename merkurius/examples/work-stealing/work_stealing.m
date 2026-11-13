% Work stealing between GPU processes, on a zoom into the Mandelbrot set.
%
% Each frame of the zoom is cut into square tiles. Some tiles are cheap,
% where the pixels escape at once, and some are costly, where they stay in
% the set until the limit, and which are which changes as the zoom goes
% deeper. So if each worker gets an equal share of the tiles, some finish
% early and wait while others still work.
%
% The workers are "places": GPU processes on one GPU, each with an equal
% share of the GPU's teams. A place computes one tile at a time, as a loop
% that runs in parallel across its teams. For each frame the host gives
% every place an equal range of tiles, and waits until all say they are done.
%
% With stealing, a place that runs out of tiles asks its neighbour in a ring
% of places for some. A place with two or more tiles left gives the thief
% the back half of them; one with fewer passes the request on to its own
% neighbour. If the request goes all the way round, the place just before
% the thief answers that there is nothing left, and the thief is done for
% the frame. Messages between places go through the host for now.
%
%   work_stealing [no-steal] [host]
%
% `no-steal` turns stealing off, so that each place keeps its own tiles, for
% comparison. `host` runs the places on the host's CPUs instead of the GPU.
% MANDEL_SIZE (2048), MANDEL_TILE (64), MANDEL_FRAMES (40) and MANDEL_PLACES
% (4) set the frames' and tiles' sides in pixels, and the counts. The
% program prints the steals, each place's tiles, and the time the zoom took.
% See README.md.

:- module work_stealing.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, float, int, list, process, require, string.

%---------------------------------------------------------------------------%
% The messages.

    % What a place is sent.
:- type to_place
    --->    neighbour(mailbox(to_place), int)
            % Its neighbour in the ring, and the neighbour's number.
    ;       frame(int, int, int)
            % Frame F, and its tiles for the place: the first, and how many.
    ;       steal(mailbox(to_place), int)
            % A request for tiles from a thief, and the thief's number.
    ;       stolen(int, int)
            % The answer to a steal: the first of the tiles, and how many.
    ;       nothing
            % The answer to a steal: no place has tiles to spare.
    ;       stop.

    % What a place sends the host.
:- type to_host
    --->    done(int, int, int, int)
            % It has no more tiles in the frame: its number, the tiles it
            % computed, their iterations, and how many of its steals got
            % tiles.
    ;       confused(int).
            % It got a message it did not expect, which is a bug: its
            % number.

%---------------------------------------------------------------------------%
% The host.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Steal = ( if list.member("no-steal", Args) then 0 else 1 ),
    Where = ( if list.member("host", Args) then host else gpu(0) ),
    setting("MANDEL_SIZE", 2048, Size, !IO),
    setting("MANDEL_TILE", 64, Tile, !IO),
    setting("MANDEL_FRAMES", 40, Frames, !IO),
    setting("MANDEL_PLACES", 4, K, !IO),
    ( if Tile >= 1, Size >= Tile, Size mod Tile = 0, Frames >= 1, K >= 1 then
        io.format("Work stealing: %d places, stealing %s\n",
            [i(K), s(if Steal = 1 then "on" else "off")], !IO),
        io.format("Frames: %d of %d x %d pixels, tiles of %d x %d\n",
            [i(Frames), i(Size), i(Size), i(Tile), i(Tile)], !IO),

        % Each place gets a K'th of the GPU's teams for its loops.
        set_worker_teams(Where, share(K), !IO),
        new_mailbox(Host, !IO),
        now(Start, !IO),
        list.map_foldl(start_place(Where, Host, Size, Tile), 0 .. K - 1,
            Places, !IO),
        tell_neighbours(Steal, Places, Places, 0, !IO),

        Tiles = (Size / Tile) * (Size / Tile),
        array.init(K, place_totals(0, 0), Stats0),
        frames(0, Frames, Host, Places, Tiles, 0, Iterations, 0, Checksum,
            Stats0, Stats, !IO),
        list.foldl(stop_place, Places, !IO),
        now(End, !IO),
        print_results(Iterations, Checksum, Stats, End - Start, !IO)
    else
        io.write_string("MANDEL_TILE must divide MANDEL_SIZE, and " ++
            "MANDEL_FRAMES and MANDEL_PLACES be at least 1.\n", !IO),
        exit_failure(!IO)
    ).

    % What the host adds up for each place over the zoom: its tiles and its
    % steals that got tiles.
:- type place_totals
    --->    place_totals(int, int).

:- pred start_place(place::in, mailbox(to_host)::in, int::in, int::in,
    int::in, mailbox(to_place)::out, io::di, io::uo) is det.
start_place(Where, Host, Size, Tile, P, Box, !IO) :-
    spawn(Where, place(Host, P, Size, Tile), Box, !IO).

    % Tells each place P of Boxes its neighbour, the next place round the
    % ring of All. With stealing off, each place is its own neighbour, so it
    % has no one to ask.
:- pred tell_neighbours(int::in, list(mailbox(to_place))::in,
    list(mailbox(to_place))::in, int::in, io::di, io::uo) is det.
tell_neighbours(_, [], _, _, !IO).
tell_neighbours(Steal, [Box | Boxes], All, P, !IO) :-
    N = ( if Steal = 1 then (P + 1) mod list.length(All) else P ),
    send(Box, neighbour(list.det_index0(All, N), N), !IO),
    tell_neighbours(Steal, Boxes, All, P + 1, !IO).

:- pred stop_place(mailbox(to_place)::in, io::di, io::uo) is det.
stop_place(Box, !IO) :-
    send(Box, stop, !IO).

    % Computes frames F to Frames - 1, adding up their iterations, a
    % checksum that weighs each frame's by its number, and each place's
    % totals.
:- pred frames(int::in, int::in, mailbox(to_host)::in,
    list(mailbox(to_place))::in, int::in, int::in, int::out, int::in, int::out,
    array(place_totals)::array_di, array(place_totals)::array_uo,
    io::di, io::uo) is det.
frames(F, Frames, Host, Places, Tiles, !Iterations, !Checksum, !Stats, !IO) :-
    ( if F < Frames then
        send_ranges(Places, 0, list.length(Places), F, Tiles, !IO),
        collect(list.length(Places), Host, 0, Frame, 0, Computed, !Stats,
            !IO),
        ( if Computed \= Tiles then
            error("work_stealing: a tile was lost or computed twice")
        else
            true
        ),
        !:Iterations = !.Iterations + Frame,
        !:Checksum = !.Checksum + (F + 1) * Frame,
        frames(F + 1, Frames, Host, Places, Tiles, !Iterations, !Checksum,
            !Stats, !IO)
    else
        true
    ).

    % Gives place P, the first of Boxes, and the rest after it, an equal
    % range of frame F's tiles, in order.
:- pred send_ranges(list(mailbox(to_place))::in, int::in, int::in, int::in,
    int::in, io::di, io::uo) is det.
send_ranges([], _, _, _, _, !IO).
send_ranges([Box | Boxes], P, K, F, Tiles, !IO) :-
    First = P * Tiles / K,
    Next = (P + 1) * Tiles / K,
    send(Box, frame(F, First, Next - First), !IO),
    send_ranges(Boxes, P + 1, K, F, Tiles, !IO).

    % Waits for the Waiting places that have not said they are done, and adds
    % up their reports.
:- pred collect(int::in, mailbox(to_host)::in, int::in, int::out,
    int::in, int::out,
    array(place_totals)::array_di, array(place_totals)::array_uo,
    io::di, io::uo) is det.
collect(Waiting, Host, !Iterations, !Tiles, !Stats, !IO) :-
    ( if Waiting > 0 then
        receive(Host, any, after(60000), R, !IO),
        ( if R = message(done(P, Tiles, Iterations, Steals)) then
            !:Iterations = !.Iterations + Iterations,
            !:Tiles = !.Tiles + Tiles,
            place_totals(T, S) = !.Stats ^ elem(P),
            !Stats ^ elem(P) := place_totals(T + Tiles, S + Steals),
            collect(Waiting - 1, Host, !Iterations, !Tiles, !Stats, !IO)
        else if R = message(confused(P)) then
            error(string.format("work_stealing: place %d got a message " ++
                "it did not expect", [i(P)]))
        else
            error("work_stealing: a place did not answer within a minute")
        )
    else
        true
    ).

    % The time is the host's, from the places' start to their last report.
:- pred print_results(int::in, int::in, array(place_totals)::in, float::in,
    io::di, io::uo) is det.
print_results(Iterations, Checksum, Stats, Time, !IO) :-
    io.format("Iterations: %d\n", [i(Iterations)], !IO),
    io.format("Checksum: %d\n", [i(Checksum)], !IO),
    Totals = array.to_list(Stats),
    io.format("Steals: %d\n", [i(list.foldl(add_steals, Totals, 0))], !IO),
    list.foldl2(print_place, Totals, 0, _, !IO),
    io.format("Time: %.3f s\n", [f(Time)], !IO).

:- pred print_place(place_totals::in, int::in, int::out, io::di, io::uo)
    is det.
print_place(place_totals(Tiles, Steals), P, P + 1, !IO) :-
    io.format("Place %d: %d tiles, %d steals\n", [i(P), i(Tiles), i(Steals)],
        !IO).

:- func add_steals(place_totals, int) = int.
add_steals(place_totals(_, Steals), Sum) = Sum + Steals.

%---------------------------------------------------------------------------%
% The places.
%
% A place moves between three states, each a predicate that calls the next
% as its last step, so that a place runs in constant stack:
% - waiting_for_frame, between frames;
% - working, while it has tiles;
% - waiting_for_tiles, after it has asked for more.
%
% Each takes the place's fixed arguments first:
%   Host        the host's mailbox, for the place's reports;
%   P           the place's number;
%   Size, Tile  the sides of a frame and of a tile, in pixels;
%   Self        the place's own mailbox;
%   Next, NextP its neighbour round the ring, and the neighbour's number.
% Then what it is doing: the frame F, and in a frame the place's counts so
% far, its tiles, their iterations and its steals. A place that gets a
% message it does not expect tells the host, which stops the program.

:- process place(mailbox(to_host), int, int, int)
    receives to_place on gpu.

place(Host, P, Size, Tile, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(neighbour(Next, NextP)) then
        waiting_for_frame(Host, P, Size, Tile, Self, Next, NextP, !IO)
    else
        send(Host, confused(P), !IO)
    ).

:- pred waiting_for_frame(mailbox(to_host)::in, int::in, int::in, int::in,
    mailbox(to_place)::in, mailbox(to_place)::in, int::in, io::di, io::uo)
    is det.
waiting_for_frame(Host, P, Size, Tile, Self, Next, NextP, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(frame(F, First, Count)) then
        working(Host, P, Size, Tile, Self, Next, NextP, F,
            First, Count, 0, 0, 0, !IO)
    else if R = message(steal(Thief, ThiefP)) then
        % A late thief: this place has nothing to give.
        pass_on(Next, NextP, Thief, ThiefP, !IO),
        waiting_for_frame(Host, P, Size, Tile, Self, Next, NextP, !IO)
    else if R = message(stop) then
        true
    else
        send(Host, confused(P), !IO)
    ).

    % Computes the place's tiles of frame F, Count of them from First, one at
    % a time, and answers any thief between two. With none left, it steals or
    % reports.
:- pred working(mailbox(to_host)::in, int::in, int::in, int::in,
    mailbox(to_place)::in, mailbox(to_place)::in, int::in, int::in,
    int::in, int::in, int::in, int::in, int::in, io::di, io::uo) is det.
working(Host, P, Size, Tile, Self, Next, NextP, F, First, Count,
        Tiles, Iterations, Steals, !IO) :-
    ( if Count > 0 then
        answer_thief(Host, P, Self, Next, NextP, First, Count, Keep, !IO),
        Sum = tile_iterations(Size, Tile, F, First),
        working(Host, P, Size, Tile, Self, Next, NextP, F,
            First + 1, Keep - 1, Tiles + 1, Iterations + Sum, Steals, !IO)
    else if NextP \= P then
        send(Next, steal(Self, P), !IO),
        waiting_for_tiles(Host, P, Size, Tile, Self, Next, NextP, F,
            Tiles, Iterations, Steals, !IO)
    else
        send(Host, done(P, Tiles, Iterations, Steals), !IO),
        waiting_for_frame(Host, P, Size, Tile, Self, Next, NextP, !IO)
    ).

    % Gives the back half of the Count tiles from First to a thief that has
    % asked, if there are two or more, and passes its request on otherwise;
    % Keep is how many tiles the place keeps. It does not wait for a thief.
:- pred answer_thief(mailbox(to_host)::in, int::in, mailbox(to_place)::in,
    mailbox(to_place)::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
answer_thief(Host, P, Self, Next, NextP, First, Count, Keep, !IO) :-
    receive(Self, any, after(0), R, !IO),
    ( if R = message(steal(Thief, ThiefP)) then
        ( if Count >= 2 then
            Give = Count / 2,
            Keep = Count - Give,
            send(Thief, stolen(First + Keep, Give), !IO)
        else
            Keep = Count,
            pass_on(Next, NextP, Thief, ThiefP, !IO)
        )
    else if R = timeout then
        Keep = Count
    else
        % The host stops the program.
        send(Host, confused(P), !IO),
        Keep = Count
    ).

    % Waits for the answer to the place's steal, passing on other thieves'.
:- pred waiting_for_tiles(mailbox(to_host)::in, int::in, int::in, int::in,
    mailbox(to_place)::in, mailbox(to_place)::in, int::in, int::in,
    int::in, int::in, int::in, io::di, io::uo) is det.
waiting_for_tiles(Host, P, Size, Tile, Self, Next, NextP, F,
        Tiles, Iterations, Steals, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(stolen(First, Count)) then
        working(Host, P, Size, Tile, Self, Next, NextP, F,
            First, Count, Tiles, Iterations, Steals + 1, !IO)
    else if R = message(nothing) then
        send(Host, done(P, Tiles, Iterations, Steals), !IO),
        waiting_for_frame(Host, P, Size, Tile, Self, Next, NextP, !IO)
    else if R = message(steal(Thief, ThiefP)) then
        pass_on(Next, NextP, Thief, ThiefP, !IO),
        waiting_for_tiles(Host, P, Size, Tile, Self, Next, NextP, F,
            Tiles, Iterations, Steals, !IO)
    else
        send(Host, confused(P), !IO)
    ).

    % Sends a thief's request on to the next place, unless that is the thief
    % itself: then every place has been asked, and the thief gets nothing.
:- pred pass_on(mailbox(to_place)::in, int::in, mailbox(to_place)::in,
    int::in, io::di, io::uo) is det.
pass_on(Next, NextP, Thief, ThiefP, !IO) :-
    ( if NextP = ThiefP then
        send(Thief, nothing, !IO)
    else
        send(Next, steal(Thief, ThiefP), !IO)
    ).

%---------------------------------------------------------------------------%
% The Mandelbrot set.

    % Frame F is 3.0 * 0.5^F wide, halving each frame; its escape limit
    % rises with it.
:- func width(int) = float.
width(F) = Width :-
    ( for(_, 1, F), fromto(3.0, W0, W1, Width) do
        W1 = W0 * 0.5
    ).

:- func limit(int) = int.
limit(F) = 256 + 64 * F.

    % The centre of the zoom, on the set's boundary. It was chosen from some
    % 300 points near the boundary as the one whose frames an equal split of
    % the tiles leaves most uneven, for a zoom of about the usual cost: with
    % four places, the busiest place's share of a frame's work is on average
    % 1.3 times the mean, against 1.14 for the much used (-0.743643887037151,
    % 0.131825904205330) of benchmarks/mandel.
:- func x0 = float.
x0 = -1.393781680564294.
:- func y0 = float.
y0 = 0.127954072006025.

    % The iterations of tile T of frame F, summed over its pixels. The loop
    % runs across the place's teams, each taking some of the pixels, and
    % the reduction adds up their sums.
:- func tile_iterations(int, int, int, int) = int.
tile_iterations(Size, Tile, F, T) = Total :-
    Step = width(F) / float(Size),
    Limit = limit(F),
    Half = Size / 2,
    PerRow = Size / Tile,
    Top = (T / PerRow) * Tile,
    Left = (T mod PerRow) * Tile,
    %$omp distribute parallel do reduction((+) : Total)
    ( for(K, 0, Tile * Tile - 1), fromto(0, S0, S1, Total) do
        % The unchecked operations skip the test for a zero Tile, which
        % main has made.
        PY = Top + unchecked_quotient(K, Tile),
        PX = Left + unchecked_rem(K, Tile),
        CX = x0 + float(PX - Half) * Step,
        CY = y0 + float(PY - Half) * Step,
        S1 = S0 + escape(CX, CY, Limit, 0, 0.0, 0.0)
    ).

    % How many iterations z = z * z + c takes to escape, from z = X + iY
    % after N of them, at most Limit.
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
% Three things from C: the clock, the environment, and exiting with an error
% status.

:- pragma foreign_decl("C", "#include <stdio.h>").
:- pragma foreign_decl("C", "#include <stdlib.h>").
:- pragma foreign_decl("C", "#include <time.h>").

    % Seconds by a clock that only goes forward.
:- pred now(float::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    T = (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
").

    % The environment variable Name's value, or Default if it is not set.
:- pred setting(string::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    setting(Name::in, Default::in, Value::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    const char *s = getenv(Name);
    Value = s && *s ? (MR_Integer) strtol(s, NULL, 10) : Default;
").

:- pred exit_failure(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    exit_failure(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    fflush(stdout);
    exit(1);
").
