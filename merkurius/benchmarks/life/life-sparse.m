% Conway's Game of Life on sparse tiles, in a GPU process: only the tiles
% where life is are stepped, starting in front of it and ending behind it
% (docs/INPROGRESS.md, sparse tiles on one place, step s5). The world is
% life.m's grid with a soup at its centre (LIFE_SOUP), dead beyond its edge,
% so the dense programs, life.m and life.c, give the same population and
% checksum for the same world and steps.
%
% - The world is LIFE_WORLD cells square, in tiles of LIFE_TILE cells
%   square, a power of two that divides it.
% - The pool: two arrays of LIFE_CAPACITY tiles' cells, uint8, the current
%   step's and the next, which swap after each step. Tile I's cells are
%   I * T * T onwards, by rows. Tile 0 is the dead tile, all zero, never
%   stepped, the neighbour of a tile where there is none.
% - The directory: a tile's pool index for each place in the world, or 0.
% - The tiles' fields, an array each (the user, 2026-11-12): the place in
%   the world; nine neighbours, by D = (DY + 1) * 3 + DX + 1 for DY and DX
%   from -1 to 1, the middle one the tile itself, so that a cell's
%   neighbour at row R and column C, each from -1 to T, is in neighbour
%   ((R >> Log) + 1) * 3 + (C >> Log) + 1, with no branch; the live cells
%   and the live edges and corners after the last step, a bit D for each
%   direction D; and the steps the tile has been quiet.
% - The active tiles' pool indices, packed, and a free list.
%
% A step is a loop under distribute over the active tiles, a team for each,
% with a loop under parallel do over its cells across the team's threads,
% which reduces its live cells and its edges' bits. After it, on the
% process's lane, the bookkeeping: a tile with no live cells and no live
% edge facing it from a neighbour for `quiet_steps` steps ends; then each
% live edge facing no tile starts one there, inside the world, linked to its
% neighbours both ways; then the new tiles' cells are zeroed, a loop across
% the teams. A tile can come alive only next to a live edge or corner of
% another, so a world without a tile somewhere is dead there.
%
% The process sends the host each step's time, the bookkeeping's part of
% it, the active tiles after each step, and the outcome. LIFE_TRACE=1 prints
% each step's line, for a chart of the active tiles over time.
%
% Built with -fprocess-declarations, -fdo-loops and --offload-arch, and
% -fopenmp for the loops to run across the teams; without -fopenmp the
% process runs them on its own lane. The first argument `host` spawns the
% process at the host, for tests. See README.md.

:- module life_sparse.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int32, list, process, string, uint16,
    uint8.

    % The live cells, the sum of their indices in the world, the most
    % active tiles at once, and whether the pool ran out of tiles.
:- type outcome
    --->    outcome(int, int, int, int).

    % The world's shape: its tiles' side and its log, the world's side in
    % tiles, and in cells.
:- type geometry
    --->    geometry(
                tile :: int,
                log :: int,
                side :: int,
                world :: int
            ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    setting("LIFE_WORLD", 8192, World, !IO),
    setting("LIFE_TILE", 64, Tile, !IO),
    setting("LIFE_SOUP", 256, Soup, !IO),
    setting("LIFE_STEPS", 1000, Steps, !IO),
    Log = log2(Tile, 0),
    Side = ( if Tile > 0 then World / Tile else 0 ),
    setting("LIFE_CAPACITY", int.min(Side * Side, 8192) + 1, Capacity, !IO),
    setting("LIFE_TRACE", 0, Trace, !IO),
    ( if
        Tile >= 4, 1 << Log = Tile,
        World >= Tile, World mod Tile = 0,
        Soup >= 0, Soup =< World,
        Steps >= 2,
        Capacity >= 2
    then
        header(World, Tile, Capacity, Soup, Steps, !IO),
        new_mailbox(TimesBox, !IO),
        new_mailbox(CountsBox, !IO),
        new_mailbox(OutcomeBox, !IO),
        spawn(Place, life(TimesBox, CountsBox, OutcomeBox, World, Log, Soup,
            Steps, Capacity), _, !IO),
        receive(TimesBox, any, after(600000), R1, !IO),
        receive(CountsBox, any, after(600000), R2, !IO),
        receive(OutcomeBox, any, after(600000), R3, !IO),
        ( if
            R1 = message(Times),
            R2 = message(Counts),
            R3 = message(outcome(Population, Checksum, Peak, Overflow))
        then
            ( if Overflow = 0 then
                ( if Trace = 0 then
                    true
                else
                    trace(Steps, Times, Counts, !IO)
                ),
                results(Steps, Times, Counts, Population, Checksum, Peak, !IO)
            else
                io.write_string("The pool ran out of tiles: " ++
                    "raise LIFE_CAPACITY.\n", !IO),
                exit_failure(!IO)
            )
        else
            io.write_string("The GPU process did not answer.\n", !IO),
            exit_failure(!IO)
        )
    else
        io.write_string("LIFE_TILE must be a power of two of at least 4 " ++
            "that divides LIFE_WORLD, LIFE_SOUP from 0 to LIFE_WORLD, " ++
            "LIFE_STEPS at least 2, and LIFE_CAPACITY at least 2.\n", !IO),
        exit_failure(!IO)
    ).

    % The log of N to base 2, if N is a power of two.
:- func log2(int, int) = int.
log2(N, L) = ( if N =< 1 then L else log2(N >> 1, L + 1) ).

%---------------------------------------------------------------------------%
% The process.

    % Steps the world Steps times and sends the host the times, the active
    % tiles after each step, and the outcome.
:- process life(mailbox(array(float)), mailbox(array(int)), mailbox(outcome),
    int, int, int, int, int) receives int on gpu.

life(TimesBox, CountsBox, OutcomeBox, World, Log, Soup, Steps, Capacity,
        _Self, !IO) :-
    T = 1 << Log,
    Side = World >> Log,
    G = geometry(T, Log, Side, World),
    array.init(Capacity * T * T, 0u8, Cur0),
    array.init(Capacity * T * T, 0u8, Next),
    array.init(Side * Side, 0i32, Dir0),
    array.init(Capacity, 0i32, Pos0),
    array.init(Capacity * 9, 0i32, Nb0),
    array.init(Capacity, 0i32, Live),
    array.init(Capacity, 0u16, Edges0),
    array.init(Capacity, 0u8, Quiet0),
    array.init(Capacity, 0i32, Actives0),
    array.init(Capacity, 0i32, Free0),
    % Tile 0 is the dead tile; the free list gives 1, 2 and so on first.
    ( for(J, 0, Capacity - 2), fromto(Free0, F0, F1, Free1) do
        F1 = F0 ^ unsafe_elem(J) := int32.cast_from_int(Capacity - 1 - J)
    ),
    % The tiles that the soup is in, its cells, and each tile's edges all
    % live, so that the tiles around them start too.
    From = (World - Soup) / 2,
    ( if Soup > 0 then
        First = From >> Log,
        Last = (From + Soup - 1) >> Log
    else
        First = 0,
        Last = -1
    ),
    soup_tiles(G, First, Last, First, Dir0, Dir1, Pos0, Pos1, Nb0, Nb1,
        Quiet0, Quiet1, Actives0, Actives1, 0, NActive1, Free1, Free2,
        Capacity - 1, NFree1, 0, Overflow1),
    init(G, From, Soup, NActive1, Actives1, Pos1, Cur0, Cur1),
    ( for(J, 0, NActive1 - 1), fromto(Edges0, E0, E1, Edges1) do
        E1 = E0 ^ unsafe_elem(int32.to_int(Actives1 ^ unsafe_elem(J))) :=
            all_edges
    ),
    grow(G, 0, NActive1, Edges1, Dir1, Dir, Pos1, Pos, Nb1, Nb,
        Quiet1, Quiet, Actives1, Actives, NActive1, NActive, Free2, Free,
        NFree1, NFree, Overflow1, Overflow),
    array.init(2 * Steps, 0.0, Times0),
    array.init(Steps, 0, Counts0),
    run(G, 0, Steps, Cur1, Next, Final, Dir, _, Pos, Pos2, Nb, _, Live,
        LiveF, Edges1, _, Quiet, _, Actives, ActivesF, NActive, NActiveF,
        Free, _, NFree, _, Overflow, OverflowF, NActive, Peak,
        Times0, Times, Counts0, Counts, !IO),
    count(G, NActiveF, ActivesF, Pos2, LiveF, Final, Population, Checksum),
    send(TimesBox, Times, !IO),
    send(CountsBox, Counts, !IO),
    send(OutcomeBox, outcome(Population, Checksum, Peak, OverflowF), !IO).

    % Every direction's bit but the middle's.
:- func all_edges = uint16.
all_edges = 0x1ef_u16.

    % The steps a tile with nothing alive near it waits before it ends, so
    % that one that life is about to cross again stays.
:- func quiet_steps = int.
quiet_steps = 4.

    % Steps from S to Steps, from Cur into Next, which swap after each, each
    % step's bookkeeping after it. Final is the cells after the last.
    %
:- pred run(geometry::in, int::in, int::in,
    array(uint8)::array_di, array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint16)::array_di, array(uint16)::array_uo,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out, int::in, int::out,
    array(float)::array_di, array(float)::array_uo,
    array(int)::array_di, array(int)::array_uo, io::di, io::uo) is det.
run(G, S, Steps, Cur, Next0, Final, !Dir, !Pos, !Nb, !Live, !Edges, !Quiet,
        !Actives, !NActive, !Free, !NFree, !Overflow, !Peak, !Times, !Counts,
        !IO) :-
    ( if S >= Steps then
        Final = Cur
    else if !.Overflow \= 0 then
        Final = Cur
    else
        now(T0, !IO),
        step(G, !.NActive, !.Actives, !.Nb, Cur, Next0, Next, !Live, !Edges),
        now(T1, !IO),
        shrink(G, 0, !.Live, !.Edges, !Dir, !.Pos, !Nb, !Quiet, !Actives,
            !NActive, !Free, !NFree),
        Kept = !.NActive,
        grow(G, 0, Kept, !.Edges, !Dir, !Pos, !Nb, !Quiet, !Actives,
            !NActive, !Free, !NFree, !Overflow),
        % A loop across the teams costs microseconds even with nothing to
        % do.
        ( if !.NActive > Kept then
            zero(G, Kept, !.NActive, !.Actives, Next, Next1)
        else
            Next1 = Next
        ),
        now(T2, !IO),
        !Times ^ elem(S) := T2 - T0,
        !Times ^ elem(Steps + S) := T2 - T1,
        !Counts ^ elem(S) := !.NActive,
        !:Peak = int.max(!.Peak, !.NActive),
        run(G, S + 1, Steps, Next1, Cur, Final, !Dir, !Pos, !Nb, !Live,
            !Edges, !Quiet, !Actives, !NActive, !Free, !NFree, !Overflow,
            !Peak, !Times, !Counts, !IO)
    ).

%---------------------------------------------------------------------------%
% The step.

    % Next's cells of the active tiles from Cur's, and each tile's live
    % cells and live edges: a team for each tile.
    %
:- pred step(geometry::in, int::in, array(int32)::array_ui,
    array(int32)::array_ui, array(uint8)::array_ui,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint16)::array_di, array(uint16)::array_uo) is det.
step(G, NActive, Actives, Nb, Cur, !Next, !Live, !Edges) :-
    %$omp distribute
    ( for(J, 0, NActive - 1) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        step_tile(G, I, Nb, Cur, !Next, Live, Edges),
        !Live ^ unsafe_elem(I) := int32.cast_from_int(Live),
        !Edges ^ unsafe_elem(I) := uint16.cast_from_int(Edges)
    ).

    % Tile I's cells, across the team's threads.
    %
:- pred step_tile(geometry::in, int::in, array(int32)::array_ui,
    array(uint8)::array_ui, array(uint8)::array_di, array(uint8)::array_uo,
    int::out, int::out) is det.
step_tile(G, I, Nb, Cur, !Next, Live, Edges) :-
    T = G ^ tile,
    Log = G ^ log,
    M = T - 1,
    Base = I * 9,
    Cells = I * T * T,
    %$omp parallel do reduction((+) : Live) reduction(\/ : Edges)
    ( for(K, 0, T * T - 1), fromto(0, L0, L1, Live),
        fromto(0, E0, E1, Edges) do
        R = K >> Log,
        C = K /\ M,
        Count =
            cell(Log, M, Base, Nb, Cur, R - 1, C - 1) +
            cell(Log, M, Base, Nb, Cur, R - 1, C) +
            cell(Log, M, Base, Nb, Cur, R - 1, C + 1) +
            cell(Log, M, Base, Nb, Cur, R, C - 1) +
            cell(Log, M, Base, Nb, Cur, R, C + 1) +
            cell(Log, M, Base, Nb, Cur, R + 1, C - 1) +
            cell(Log, M, Base, Nb, Cur, R + 1, C) +
            cell(Log, M, Base, Nb, Cur, R + 1, C + 1),
        ( if Count = 3u8 then
            V = 1u8
        else if Count = 2u8 then
            V = Cur ^ unsafe_elem(Cells + K)
        else
            V = 0u8
        ),
        !Next ^ unsafe_elem(Cells + K) := V,
        ( if V = 1u8 then
            L1 = L0 + 1,
            E1 = E0 \/ edge_bits(R, C, M)
        else
            L1 = L0,
            E1 = E0
        )
    ).

    % The cell at row R and column C of the tile whose neighbours start at
    % Base, each from -1 to the tile's side: in the tile, or in the
    % neighbour that R and C reach, at the row and column there.
    %
:- func cell(int, int, int, array(int32), array(uint8), int, int) = uint8.
:- mode cell(in, in, in, array_ui, array_ui, in, in) = out is det.
cell(Log, M, Base, Nb, Cur, R, C) = Cur ^ unsafe_elem(Cell) :-
    D = ((R >> Log) + 1) * 3 + (C >> Log) + 1,
    Tile = int32.to_int(Nb ^ unsafe_elem(Base + D)),
    Cell = (Tile << (2 * Log)) + ((R /\ M) << Log) + (C /\ M).

    % The directions that a live cell at row R and column C makes live: the
    % edge it is on, if any, each, and the corner.
    %
:- func edge_bits(int, int, int) = int.
edge_bits(R, C, M) = bit(Row, 0) \/ bit(0, Column) \/ bit(Row, Column) :-
    Row = ( if R = 0 then -1 else if R = M then 1 else 0 ),
    Column = ( if C = 0 then -1 else if C = M then 1 else 0 ).

:- func bit(int, int) = int.
bit(DY, DX) = ( if DY = 0, DX = 0 then 0 else 1 << ((DY + 1) * 3 + DX + 1) ).

%---------------------------------------------------------------------------%
% The bookkeeping, on the process's lane.

    % From the J'th active tile on: each tile with no live cells, and no
    % neighbour with a live edge facing it, for quiet_steps steps ends,
    % its place in the actives taken by the last one.
    %
:- pred shrink(geometry::in, int::in, array(int32)::array_ui,
    array(uint16)::array_ui, array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_ui, array(int32)::array_di, array(int32)::array_uo,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out) is det.
shrink(G, J, Live, Edges, !Dir, Pos, !Nb, !Quiet, !Actives, !NActive, !Free,
        !NFree) :-
    ( if J >= !.NActive then
        true
    else
        I = int32.to_int(!.Actives ^ unsafe_elem(J)),
        ( if
            Live ^ unsafe_elem(I) = 0i32,
            not facing_edge(I, 0, !.Nb, Edges)
        then
            Q = uint8.to_int(!.Quiet ^ unsafe_elem(I)) + 1,
            ( if Q >= quiet_steps then
                unlink(I, 0, !Nb),
                !Dir ^ unsafe_elem(int32.to_int(Pos ^ unsafe_elem(I))) := 0i32,
                !Free ^ unsafe_elem(!.NFree) := int32.cast_from_int(I),
                !:NFree = !.NFree + 1,
                !:NActive = !.NActive - 1,
                !Actives ^ unsafe_elem(J) :=
                    !.Actives ^ unsafe_elem(!.NActive),
                shrink(G, J, Live, Edges, !Dir, Pos, !Nb, !Quiet, !Actives,
                    !NActive, !Free, !NFree)
            else
                !Quiet ^ unsafe_elem(I) := uint8.cast_from_int(Q),
                shrink(G, J + 1, Live, Edges, !Dir, Pos, !Nb, !Quiet,
                    !Actives, !NActive, !Free, !NFree)
            )
        else
            !Quiet ^ unsafe_elem(I) := 0u8,
            shrink(G, J + 1, Live, Edges, !Dir, Pos, !Nb, !Quiet, !Actives,
                !NActive, !Free, !NFree)
        )
    ).

    % Whether a neighbour of tile I, in direction D or after it, has a live
    % edge or corner facing I.
    %
:- pred facing_edge(int::in, int::in, array(int32)::array_ui,
    array(uint16)::array_ui) is semidet.
facing_edge(I, D, Nb, Edges) :-
    D =< 8,
    ( if D = 4 then
        Neighbour = 0
    else
        Neighbour = int32.to_int(Nb ^ unsafe_elem(I * 9 + D))
    ),
    ( if
        Neighbour \= 0,
        uint16.to_int(Edges ^ unsafe_elem(Neighbour)) /\ (1 << (8 - D)) \= 0
    then
        true
    else
        facing_edge(I, D + 1, Nb, Edges)
    ).

    % Tile I's neighbours from direction D on forget it.
    %
:- pred unlink(int::in, int::in, array(int32)::array_di,
    array(int32)::array_uo) is det.
unlink(I, D, !Nb) :-
    ( if D > 8 then
        true
    else
        Neighbour = int32.to_int(!.Nb ^ unsafe_elem(I * 9 + D)),
        ( if D \= 4, Neighbour \= 0 then
            !Nb ^ unsafe_elem(Neighbour * 9 + 8 - D) := 0i32
        else
            true
        ),
        unlink(I, D + 1, !Nb)
    ).

    % From the J'th active tile to the Kept'th: each live edge or corner
    % that faces no tile, inside the world, starts one there.
    %
:- pred grow(geometry::in, int::in, int::in, array(uint16)::array_ui,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out) is det.
grow(G, J, Kept, Edges, !Dir, !Pos, !Nb, !Quiet, !Actives, !NActive, !Free,
        !NFree, !Overflow) :-
    ( if J >= Kept then
        true
    else
        I = int32.to_int(!.Actives ^ unsafe_elem(J)),
        grow_tile(G, I, 0, uint16.to_int(Edges ^ unsafe_elem(I)), !Dir, !Pos,
            !Nb, !Quiet, !Actives, !NActive, !Free, !NFree, !Overflow),
        grow(G, J + 1, Kept, Edges, !Dir, !Pos, !Nb, !Quiet, !Actives,
            !NActive, !Free, !NFree, !Overflow)
    ).

    % Tile I's directions from D on, of its live edges' bits Bits.
    %
:- pred grow_tile(geometry::in, int::in, int::in, int::in,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out) is det.
grow_tile(G, I, D, Bits, !Dir, !Pos, !Nb, !Quiet, !Actives, !NActive, !Free,
        !NFree, !Overflow) :-
    ( if D > 8 then
        true
    else
        ( if
            Bits /\ (1 << D) \= 0,
            !.Nb ^ unsafe_elem(I * 9 + D) = 0i32,
            beside(G, int32.to_int(!.Pos ^ unsafe_elem(I)), D, Place)
        then
            start(G, Place, !Dir, !Pos, !Nb, !Quiet, !Actives, !NActive,
                !Free, !NFree, !Overflow)
        else
            true
        ),
        grow_tile(G, I, D + 1, Bits, !Dir, !Pos, !Nb, !Quiet, !Actives,
            !NActive, !Free, !NFree, !Overflow)
    ).

    % The place in direction D of place P, if it is in the world.
    %
:- pred beside(geometry::in, int::in, int::in, int::out) is semidet.
beside(G, P, D, Place) :-
    Side = G ^ side,
    Y = P / Side + D / 3 - 1,
    X = P mod Side + D mod 3 - 1,
    Y >= 0, Y < Side,
    X >= 0, X < Side,
    Place = Y * Side + X.

    % A tile at Place, from the free list, linked to its neighbours both
    % ways and active; or Overflow set, if there is none free.
    %
:- pred start(geometry::in, int::in,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out) is det.
start(G, Place, !Dir, !Pos, !Nb, !Quiet, !Actives, !NActive, !Free, !NFree,
        !Overflow) :-
    ( if !.NFree = 0 then
        !:Overflow = 1
    else
        !:NFree = !.NFree - 1,
        I = int32.to_int(!.Free ^ unsafe_elem(!.NFree)),
        !Dir ^ unsafe_elem(Place) := int32.cast_from_int(I),
        !Pos ^ unsafe_elem(I) := int32.cast_from_int(Place),
        !Quiet ^ unsafe_elem(I) := 0u8,
        link(G, I, Place, 0, !.Dir, !Nb),
        !Actives ^ unsafe_elem(!.NActive) := int32.cast_from_int(I),
        !:NActive = !.NActive + 1
    ).

    % Tile I's neighbours from direction D on, from the directory, and
    % theirs that face it.
    %
:- pred link(geometry::in, int::in, int::in, int::in, array(int32)::array_ui,
    array(int32)::array_di, array(int32)::array_uo) is det.
link(G, I, Place, D, Dir, !Nb) :-
    ( if D > 8 then
        true
    else
        ( if D = 4 then
            Neighbour = I
        else if beside(G, Place, D, There) then
            Neighbour = int32.to_int(Dir ^ unsafe_elem(There))
        else
            Neighbour = 0
        ),
        !Nb ^ unsafe_elem(I * 9 + D) := int32.cast_from_int(Neighbour),
        ( if D \= 4, Neighbour \= 0 then
            !Nb ^ unsafe_elem(Neighbour * 9 + 8 - D) := int32.cast_from_int(I)
        else
            true
        ),
        link(G, I, Place, D + 1, Dir, !Nb)
    ).

    % The tiles from row Y and column X to row and column Last, each from
    % column First, by rows.
    %
:- pred soup_tiles(geometry::in, int::in, int::in, int::in,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out) is det.
soup_tiles(G, Y, Last, First, !Dir, !Pos, !Nb, !Quiet, !Actives, !NActive,
        !Free, !NFree, !Overflow) :-
    ( if Y > Last then
        true
    else
        ( for(X, First, Last), fromto(!.Dir, D0, D1, Dir1),
            fromto(!.Pos, P0, P1, Pos1), fromto(!.Nb, N0, N1, Nb1),
            fromto(!.Quiet, Q0, Q1, Quiet1), fromto(!.Actives, A0, A1, Act1),
            fromto(!.NActive, C0, C1, NActive1),
            fromto(!.Free, F0, F1, Free1), fromto(!.NFree, R0, R1, NFree1),
            fromto(!.Overflow, O0, O1, Overflow1) do
            start(G, Y * G ^ side + X, D0, D1, P0, P1, N0, N1, Q0, Q1,
                A0, A1, C0, C1, F0, F1, R0, R1, O0, O1)
        ),
        soup_tiles(G, Y + 1, Last, First, Dir1, !:Dir, Pos1, !:Pos, Nb1, !:Nb,
            Quiet1, !:Quiet, Act1, !:Actives, NActive1, !:NActive, Free1,
            !:Free, NFree1, !:NFree, Overflow1, !:Overflow)
    ).

%---------------------------------------------------------------------------%
% Loops across the teams: the soup's cells, new tiles' zeros, and the count.

    % Whether the cell of index K in the world starts alive, if it is in
    % the soup: about 30% do, by a multiplicative hash of K, as in life.m.
    %
:- func alive(int) = uint8.
alive(K) = V :-
    H = (K * 2654435761) /\ 4294967295,
    ( if unchecked_rem(H >> 16, 100) < 30 then V = 1u8 else V = 0u8 ).

    % The active tiles' cells at the start: those in the soup, whose rows
    % and columns are From to From + Soup - 1, alive by alive/1.
    %
:- pred init(geometry::in, int::in, int::in, int::in, array(int32)::array_ui,
    array(int32)::array_ui, array(uint8)::array_di, array(uint8)::array_uo)
    is det.
init(G, From, Soup, NActive, Actives, Pos, !Cur) :-
    %$omp distribute
    ( for(J, 0, NActive - 1) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        init_tile(G, From, Soup, I, int32.to_int(Pos ^ unsafe_elem(I)), !Cur)
    ).

:- pred init_tile(geometry::in, int::in, int::in, int::in, int::in,
    array(uint8)::array_di, array(uint8)::array_uo) is det.
init_tile(G, From, Soup, I, P, !Cur) :-
    T = G ^ tile,
    Log = G ^ log,
    Top = (P / G ^ side) << Log,
    Left = (P mod G ^ side) << Log,
    World = G ^ world,
    %$omp parallel do
    ( for(K, 0, T * T - 1) do
        Row = Top + (K >> Log),
        Column = Left + (K /\ (T - 1)),
        ( if
            Row >= From, Row < From + Soup,
            Column >= From, Column < From + Soup
        then
            V = alive(Row * World + Column)
        else
            V = 0u8
        ),
        !Cur ^ unsafe_elem((I << (2 * Log)) + K) := V
    ).

    % The cells of the active tiles from the First'th on, which are new, all
    % dead, in the cells that the next step reads.
    %
:- pred zero(geometry::in, int::in, int::in, array(int32)::array_ui,
    array(uint8)::array_di, array(uint8)::array_uo) is det.
zero(G, First, NActive, Actives, !Cells) :-
    %$omp distribute
    ( for(J, First, NActive - 1) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        zero_tile(G, I, !Cells)
    ).

:- pred zero_tile(geometry::in, int::in,
    array(uint8)::array_di, array(uint8)::array_uo) is det.
zero_tile(G, I, !Cells) :-
    Log = G ^ log,
    T = G ^ tile,
    %$omp parallel do
    ( for(K, 0, T * T - 1) do
        !Cells ^ unsafe_elem((I << (2 * Log)) + K) := 0u8
    ).

    % The live cells, from the last step's counts, and the sum of their
    % indices in the world, which says where they are, from each tile's
    % across its team.
    %
:- pred count(geometry::in, int::in, array(int32)::array_ui,
    array(int32)::array_ui, array(int32)::array_ui, array(uint8)::array_ui,
    int::out, int::out) is det.
count(G, NActive, Actives, Pos, Live, Cells, Population, Checksum) :-
    array.init(int.max(NActive, 1), 0, Sums0),
    %$omp distribute
    ( for(J, 0, NActive - 1), fromto(Sums0, S0, S1, Sums) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        tile_checksum(G, I, int32.to_int(Pos ^ unsafe_elem(I)), Cells, Sum),
        S1 = S0 ^ unsafe_elem(J) := Sum
    ),
    ( for(J, 0, NActive - 1), fromto(0, P0, P1, Population),
        fromto(0, C0, C1, Checksum) do
        P1 = P0 + int32.to_int(Live ^ unsafe_elem(
            int32.to_int(Actives ^ unsafe_elem(J)))),
        C1 = C0 + Sums ^ unsafe_elem(J)
    ).

:- pred tile_checksum(geometry::in, int::in, int::in, array(uint8)::array_ui,
    int::out) is det.
tile_checksum(G, I, P, Cells, Sum) :-
    T = G ^ tile,
    Log = G ^ log,
    Top = (P / G ^ side) << Log,
    Left = (P mod G ^ side) << Log,
    World = G ^ world,
    %$omp parallel do reduction((+) : Sum)
    ( for(K, 0, T * T - 1), fromto(0, S0, S1, Sum) do
        V = uint8.to_int(Cells ^ unsafe_elem((I << (2 * Log)) + K)),
        S1 = S0 + V * ((Top + (K >> Log)) * World + Left + (K /\ (T - 1)))
    ).

%---------------------------------------------------------------------------%
% The host's output.

    % Each step's active tiles and time, a line each.
    %
:- pred trace(int::in, array(float)::in, array(int)::in, io::di, io::uo)
    is det.
trace(Steps, Times, Counts, !IO) :-
    ( for(S, 0, Steps - 1), fromto(!.IO, IO0, IO1, !:IO) do
        trace_line(S, Counts ^ elem(S), Times ^ elem(S),
            Times ^ elem(Steps + S), IO0, IO1)
    ).

    % The time of each step but the first, which includes the process's
    % first loops across its teams, as min, max and average, with the
    % bookkeeping's part of the average; the world's cells a second from
    % the average step, as dense programs would have to step them; and the
    % active tiles.
    %
:- pred results(int::in, array(float)::in, array(int)::in, int::in, int::in,
    int::in, io::di, io::uo) is det.
results(Steps, Times, Counts, Population, Checksum, Peak, !IO) :-
    ( for(S, 1, Steps - 1),
        fromto(Times ^ elem(1), Min0, Min1, Min),
        fromto(Times ^ elem(1), Max0, Max1, Max),
        fromto(0.0, Sum0, Sum1, Sum),
        fromto(0.0, Keep0, Keep1, Keep) do
        Min1 = float.min(Min0, Times ^ elem(S)),
        Max1 = float.max(Max0, Times ^ elem(S)),
        Sum1 = Sum0 + Times ^ elem(S),
        Keep1 = Keep0 + Times ^ elem(Steps + S)
    ),
    Average = Sum / float(Steps - 1),
    write_results(Population, Checksum, Peak, Counts ^ elem(Steps - 1), Min,
        Max, Average, Keep / float(Steps - 1), !IO).

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

:- pred header(int::in, int::in, int::in, int::in, int::in, io::di, io::uo)
    is det.
:- pragma foreign_proc("C",
    header(World::in, Tile::in, Capacity::in, Soup::in, Steps::in,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Game of Life, Mercury, GPU process, sparse tiles\\n\");
    printf(\"World: %ld x %ld cells, tiles of %ld x %ld, %ld in the pool\\n\",
        (long) World, (long) World, (long) Tile, (long) Tile,
        (long) Capacity - 1);
    printf(\"Soup: %ld x %ld cells at the centre\\n\", (long) Soup,
        (long) Soup);
    printf(\"Steps: %ld\\n\", (long) Steps);
").

:- pred trace_line(int::in, int::in, float::in, float::in, io::di, io::uo)
    is det.
:- pragma foreign_proc("C",
    trace_line(S::in, Active::in, Time::in, Keep::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"step %ld: %ld tiles, %.6f s, bookkeeping %.6f s\\n\", (long) S,
        (long) Active, Time, Keep);
").

:- pred write_results(int::in, int::in, int::in, int::in, float::in,
    float::in, float::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Population::in, Checksum::in, Peak::in, Final::in, Min::in,
        Max::in, Average::in, Keep::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Population: %ld\\n\", (long) Population);
    printf(\"Checksum: %ld\\n\", (long) Checksum);
    printf(\"Active tiles: %ld at most, %ld at the end\\n\", (long) Peak,
        (long) Final);
    printf(\"%-12s%-12s%-12s%-14s\\n\", \"Min (sec)\", \"Max\", \"Average\",
        \"Bookkeeping\");
    printf(\"%-12.6f%-12.6f%-12.6f%-14.6f\\n\", Min, Max, Average, Keep);
").
