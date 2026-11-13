% Conway's Game of Life on sparse tiles, as life-sparse.m, run by several
% places on one GPU, each a GPU process with a share of the GPU's teams
% that owns some of the tiles (docs/INPROGRESS.md, step 5 of the demo,
% several places with a directory). Its population, checksum and active
% tiles are life-sparse.m's for the same world and steps, for any number of
% places. Tiles have fixed owners: the directory places each new tile and
% never moves it.
%
% - The directory is the program's main, on the host. It holds the world
%   map, the place that owns each tile position or none, and alone changes
%   it. The soup's tiles go round the places by rows; a new tile goes to the
%   place whose tile asked for it, the first by place number if several
%   did.
% - A place keeps life-sparse.m's pool, directory and neighbour links, with
%   two kinds of tile in its pool: its own, which it steps, and ghosts, a
%   copy of each tile that another place owns next to one of its own, never
%   stepped, whose border cells arrive each step. A cell at a tile's edge
%   reads a ghost through the neighbour links as it reads its own tiles, so
%   the step is life-sparse.m's. It also keeps a copy of the world map.
%
% A step, s to s + 1, for each place:
% 1. Step its own tiles, with the ghosts holding step s's borders.
% 2. Send the directory a summary of each of its tiles: its position,
%    whether it has live cells, and its live edges and corners.
% 3. The directory decides every tile's fate, as life-sparse.m's `decide`
%    does, from all the places' summaries, and sends every place the
%    changes to the map.
% 4. Each place applies them: it ends its tiles that end, starts the new
%    ones given to it, all dead, and drops the ghosts it no longer needs.
%    It then sends each place that owns a tile next to one of its own a
%    halo: the border cells of each of its tiles next to that place's, each
%    after its position.
% 5. It takes a halo from each such place, which may come before the map,
%    and copies its cells into the ghosts, starting a ghost for a tile it
%    has none of.
%
% The directory decides, rather than each place, since a place's tile ends
% only if no neighbour's live edge faces it after the step, and a
% neighbour's edges are known to its owner alone until its halo comes.
% Halos carry their tiles' positions, since a place may get a neighbour's
% halo before the directory's map, and a GPU process takes its messages in
% the order they come.
%
% LIFE_WORLD, LIFE_TILE, LIFE_SOUP, LIFE_STEPS and LIFE_CAPACITY are
% life-sparse.m's, LIFE_CAPACITY for each place's pool; LIFE_PLACES, 2 by
% default, is the number of places. Built with -fprocess-declarations,
% -fdo-loops and --offload-arch, and -fopenmp for the loops to run across
% each place's teams. The first argument `host` spawns the places at the
% host, for tests. See README.md.

:- module life_places.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int32, list, process, string, uint16,
    uint8.

    % The world's shape: its tiles' side and its log, the world's side in
    % tiles, and in cells.
:- type geometry
    --->    geometry(
                tile :: int,
                log :: int,
                side :: int,
                world :: int
            ).

    % What a place receives:
    % - peer(J, Mailbox): place J's mailbox, before anything else;
    % - map(Step, Changes): the changes to the world map after Step, or the
    %   tiles at the start for step -1, each Position << 8 \/ Owner, Owner
    %   the place's number plus one, or 0 for a tile that ends;
    % - halo(Place, Step, Cells): from that place, after Step, for each of
    %   its tiles next to the receiver's: the tile's position in 4 bytes,
    %   least significant first, then its 4T - 4 border cells (ring/3).
:- type to_place
    --->    peer(int, mailbox(to_place))
    ;       map(int32, array(int32))
    ;       halo(int, int32, array(uint8)).

    % What the directory receives:
    % - ready(Place): the place has all its peers;
    % - summary(Place, Step, Trouble, Tiles): after Step, for each of the
    %   place's tiles, Position << 10 \/ Dead << 9 \/ Edges, Dead 1 if it
    %   has no live cells, Edges its live edges and corners as
    %   life-sparse.m's; Trouble as below;
    % - outcome(Place, Population, Checksum, Trouble): at the end, its
    %   tiles' live cells and the sum of their indices in the world. Trouble
    %   is 0, or 1 if the place's pool ran out of tiles, or 2 if a message
    %   came that the protocol has no place for.
:- type to_directory
    --->    ready(int)
    ;       summary(int, int32, int, array(int32))
    ;       outcome(int, int, int, int).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Where = ( if Args = ["host"] then host else gpu(0) ),
    setting("LIFE_WORLD", 8192, World, !IO),
    setting("LIFE_TILE", 64, Tile, !IO),
    setting("LIFE_SOUP", 256, Soup, !IO),
    setting("LIFE_STEPS", 1000, Steps, !IO),
    setting("LIFE_PLACES", 2, N, !IO),
    Log = log2(Tile, 0),
    Side = ( if Tile > 0 then World / Tile else 0 ),
    setting("LIFE_CAPACITY", int.min(Side * Side, 8192) + 1, Capacity, !IO),
    ( if
        Tile >= 4, 1 << Log = Tile,
        World >= Tile, World mod Tile = 0,
        Side * Side < 1 << 21,
        Soup >= 0, Soup =< World,
        Steps >= 2,
        Capacity >= 2,
        N >= 1, N =< 32
    then
        header(World, Tile, Capacity, Soup, Steps, N, !IO),
        G = geometry(Tile, Log, Side, World),
        set_worker_teams(Where, share(N), !IO),
        new_mailbox(Me, !IO),
        list.map_foldl(spawn_place(Where, Me, N, World, Log, Soup, Steps,
            Capacity), 0 .. N - 1, Places, !IO),
        list.foldl(introduce(Places), Places, !IO),
        readies(Me, N, !IO),
        directory(G, Soup, Steps, N, Places, Me, !IO)
    else
        io.write_string("LIFE_TILE must be a power of two of at least 4 " ++
            "that divides LIFE_WORLD, with fewer than 2^21 tiles, " ++
            "LIFE_SOUP from 0 to LIFE_WORLD, LIFE_STEPS at least 2, " ++
            "LIFE_CAPACITY at least 2, and LIFE_PLACES from 1 to 32.\n", !IO),
        io.set_exit_status(1, !IO)
    ).

    % The log of N to base 2, if N is a power of two.
:- func log2(int, int) = int.
log2(N, L) = ( if N =< 1 then L else log2(N >> 1, L + 1) ).

:- pred spawn_place(place::in, mailbox(to_directory)::in, int::in, int::in,
    int::in, int::in, int::in, int::in, int::in, mailbox(to_place)::out,
    io::di, io::uo) is det.
spawn_place(Where, Me, N, World, Log, Soup, Steps, Capacity, J, Place,
        !IO) :-
    spawn(Where, place(Me, J, N, World, Log, Soup, Steps, Capacity), Place,
        !IO).

    % Sends Place every place's mailbox.
:- pred introduce(list(mailbox(to_place))::in, mailbox(to_place)::in,
    io::di, io::uo) is det.
introduce(Places, Place, !IO) :-
    introduce_from(0, Places, Place, !IO).

:- pred introduce_from(int::in, list(mailbox(to_place))::in,
    mailbox(to_place)::in, io::di, io::uo) is det.
introduce_from(_, [], _, !IO).
introduce_from(J, [M | Ms], Place, !IO) :-
    send(Place, peer(J, M), !IO),
    introduce_from(J + 1, Ms, Place, !IO).

    % Waits until N places have all their peers, so that no halo can come
    % before a place knows its peers.
:- pred readies(mailbox(to_directory)::in, int::in, io::di, io::uo) is det.
readies(Me, Left, !IO) :-
    ( if Left > 0 then
        receive(Me, any, after(600000), R, !IO),
        ( if R = message(ready(_)) then
            readies(Me, Left - 1, !IO)
        else
            io.write_string("A place did not answer.\n", !IO),
            exit_failure(!IO)
        )
    else
        true
    ).

%---------------------------------------------------------------------------%
% The directory.

    % The steps that a tile with nothing alive near it waits before it ends,
    % life-sparse.m's quiet_steps.
:- func quiet_steps = int.
quiet_steps = 4.

    % Places the soup's tiles and those around them, then runs the steps:
    % each step's summaries, the changes, and their time; then the outcome.
:- pred directory(geometry::in, int::in, int::in, int::in,
    list(mailbox(to_place))::in, mailbox(to_directory)::in,
    io::di, io::uo) is det.
directory(G, Soup, Steps, N, Places, Me, !IO) :-
    Side = G ^ side,
    Log = G ^ log,
    array.init(Side * Side, 0, Owner0),
    array.init(Side * Side, 0, Quiet),
    array.init(Side * Side, 0, EdgeAt),
    From = (G ^ world - Soup) / 2,
    ( if Soup > 0 then
        First = From >> Log,
        Last = (From + Soup - 1) >> Log
    else
        First = 0,
        Last = -1
    ),
    soup_owners(G, N, First, Last, First, Owner0, Owner1, [], Soups0),
    Soups = list.reverse(Soups0),
    grow_all(G, Soups, Owner1, Owner, [], New0),
    Changes0 = Soups ++ list.reverse(New0),
    broadcast(Places, map(-1i32, changes(Changes0)), !IO),
    now(T0, !IO),
    array.init(Steps, 0.0, Times0),
    array.init(Steps, 0, Counts0),
    steps(G, 0, Steps, N, Places, Me, list.length(Changes0), 0, Peak, 0,
        Trouble0, T0, Owner, _, Quiet, _, EdgeAt, _, Times0, Times,
        Counts0, Counts, !IO),
    outcomes(Me, N, 0, Population, 0, Checksum, Trouble0, Trouble, !IO),
    ( if Trouble = 0 then
        results(Steps, Times, Population, Checksum, Peak,
            Counts ^ elem(Steps - 1), !IO)
    else if Trouble = 1 then
        io.write_string("A place's pool ran out of tiles: " ++
            "raise LIFE_CAPACITY.\n", !IO),
        io.set_exit_status(1, !IO)
    else
        io.write_string("A place got a message out of turn.\n", !IO),
        io.set_exit_status(1, !IO)
    ).

    % The soup's tiles from row Y on, each from column First to Last, by
    % rows, row Y's owner place (Y - First) mod N, numbered from 1; Soups
    % their changes, the last first.
:- pred soup_owners(geometry::in, int::in, int::in, int::in, int::in,
    array(int)::array_di, array(int)::array_uo, list(int)::in,
    list(int)::out) is det.
soup_owners(G, N, Y, Last, First, !Owner, !Soups) :-
    ( if Y > Last then
        true
    else
        ( for(X, First, Last), fromto(!.Owner, O0, O1, Owner1),
            fromto(!.Soups, S0, [(P << 8) \/ Owner | S0], Soups1) do
            P = Y * G ^ side + X,
            Owner = (Y - First) mod N + 1,
            O1 = O0 ^ elem(P) := Owner
        ),
        soup_owners(G, N, Y + 1, Last, First, Owner1, !:Owner, Soups1,
            !:Soups)
    ).

    % For each of the soup's tiles in turn, from its change, a tile in each
    % direction where
    % there is none and that is in the world, as life-sparse.m starts them,
    % the soup tile's owner's; New has their changes, the last first.
:- pred grow_all(geometry::in, list(int)::in,
    array(int)::array_di, array(int)::array_uo, list(int)::in,
    list(int)::out) is det.
grow_all(_, [], !Owner, !New).
grow_all(G, [C | Cs], !Owner, !New) :-
    P = C >> 8,
    Asker = C /\ 0xff,
    ( for(D, 0, 8), fromto(!.Owner, O0, O1, !:Owner),
        fromto(!.New, N0, N1, !:New) do
        ( if
            D \= 4,
            beside(G, P, D, There),
            O0 ^ elem(There) = 0
        then
            O1 = O0 ^ elem(There) := Asker,
            N1 = [(There << 8) \/ Asker | N0]
        else
            O1 = O0,
            N1 = N0
        )
    ),
    grow_all(G, Cs, !Owner, !New).

    % Steps from S to Steps: each step's summaries, every tile's fate from
    % them, and the changes to every place. Count is the tiles in the world.
:- pred steps(geometry::in, int::in, int::in, int::in,
    list(mailbox(to_place))::in, mailbox(to_directory)::in, int::in,
    int::in, int::out, int::in, int::out, float::in,
    array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(int)::array_di, array(int)::array_uo, io::di, io::uo) is det.
steps(G, S, Steps, N, Places, Me, Count, !Peak, !Trouble, T0, !Owner,
        !Quiet, !EdgeAt, !Times, !Counts, !IO) :-
    ( if S >= Steps then
        true
    else
        summaries(Me, S, N, [], Summaries, !Trouble, !IO),
        Tiles = list.condense(list.map(summary_tiles,
            list.sort(Summaries))),
        note_edges(Tiles, !EdgeAt),
        decide(G, Tiles, !.Owner, !.EdgeAt, !Quiet, [], Ended, [], Wanted),
        end_tiles(Ended, !Owner),
        want(list.reverse(Wanted), !Owner, !Quiet, [], New),
        Changes = list.map(ended, Ended) ++ list.reverse(New),
        broadcast(Places, map(int32.cast_from_int(S), changes(Changes)), !IO),
        now(T1, !IO),
        Count1 = Count - list.length(Ended) + list.length(New),
        !Times ^ elem(S) := T1 - T0,
        !Counts ^ elem(S) := Count1,
        !:Peak = int.max(!.Peak, Count1),
        steps(G, S + 1, Steps, N, Places, Me, Count1, !Peak, !Trouble, T1,
            !Owner, !Quiet, !EdgeAt, !Times, !Counts, !IO)
    ).

    % The summaries of step S from Left places, each with its place.
:- pred summaries(mailbox(to_directory)::in, int::in, int::in,
    list({int, array(int32)})::in, list({int, array(int32)})::out,
    int::in, int::out, io::di, io::uo) is det.
summaries(Me, S, Left, !Summaries, !Trouble, !IO) :-
    ( if Left > 0 then
        receive(Me, any, after(600000), R, !IO),
        ( if
            R = message(summary(Place, Step, Trouble, Tiles)),
            int32.to_int(Step) = S
        then
            !:Summaries = [{Place, Tiles} | !.Summaries],
            !:Trouble = int.max(!.Trouble, Trouble),
            summaries(Me, S, Left - 1, !Summaries, !Trouble, !IO)
        else
            io.format("A place did not send its summary of step %d.\n",
                [i(S)], !IO),
            exit_failure(!IO)
        )
    else
        true
    ).

:- func summary_tiles({int, array(int32)}) = list(int).
summary_tiles({_, Tiles}) = list.map(int32.to_int, array.to_list(Tiles)).

:- func tile_place(int) = int.
tile_place(Tile) = Tile >> 10.

:- func tile_dead(int) = int.
tile_dead(Tile) = (Tile >> 9) /\ 1.

:- func tile_edges(int) = int.
tile_edges(Tile) = Tile /\ 0x1ff.

:- pred note_edges(list(int)::in, array(int)::array_di,
    array(int)::array_uo) is det.
note_edges([], !EdgeAt).
note_edges([Tile | Tiles], !EdgeAt) :-
    !EdgeAt ^ elem(tile_place(Tile)) := tile_edges(Tile),
    note_edges(Tiles, !EdgeAt).

    % Each tile's fate, as life-sparse.m's decide: its quiet steps, which
    % end it at quiet_steps, and the positions where it starts a tile, each
    % with its owner, the last first.
:- pred decide(geometry::in, list(int)::in, array(int)::array_ui,
    array(int)::array_ui,
    array(int)::array_di, array(int)::array_uo, list(int)::in,
    list(int)::out, list({int, int})::in, list({int, int})::out) is det.
decide(_, [], _, _, !Quiet, !Ended, !Wanted).
decide(G, [Tile | Tiles], Owner, EdgeAt, !Quiet, !Ended, !Wanted) :-
    P = tile_place(Tile),
    ( if
        tile_dead(Tile) = 1,
        facing(G, Owner, EdgeAt, P) = 0
    then
        Q = !.Quiet ^ unsafe_elem(P) + 1
    else
        Q = 0
    ),
    !Quiet ^ elem(P) := Q,
    ( if Q >= quiet_steps then
        !:Ended = [P | !.Ended]
    else
        true
    ),
    Starts = tile_edges(Tile) /\ missing(G, Owner, P),
    Asker = Owner ^ unsafe_elem(P),
    ( for(D, 0, 8), fromto(!.Wanted, W0, W1, !:Wanted) do
        ( if
            Starts /\ (1 << D) \= 0,
            beside(G, P, D, There)
        then
            W1 = [{There, Asker} | W0]
        else
            W1 = W0
        )
    ),
    decide(G, Tiles, Owner, EdgeAt, !Quiet, !Ended, !Wanted).

    % The directions in which the tile at P has a neighbour with a live
    % edge or corner facing it.
:- func facing(geometry, array(int), array(int), int) = int.
:- mode facing(in, array_ui, array_ui, in) = out is det.
facing(G, Owner, EdgeAt, P) = Facing :-
    ( for(D, 0, 8), fromto(0, F0, F1, Facing) do
        ( if
            D \= 4,
            beside(G, P, D, There),
            Owner ^ unsafe_elem(There) \= 0,
            EdgeAt ^ unsafe_elem(There) /\ (1 << (8 - D)) \= 0
        then
            F1 = F0 \/ (1 << D)
        else
            F1 = F0
        )
    ).

    % The directions from P, in the world, where there is no tile.
:- func missing(geometry, array(int), int) = int.
:- mode missing(in, array_ui, in) = out is det.
missing(G, Owner, P) = Missing :-
    ( for(D, 0, 8), fromto(0, M0, M1, Missing) do
        ( if
            D \= 4,
            beside(G, P, D, There),
            Owner ^ unsafe_elem(There) = 0
        then
            M1 = M0 \/ (1 << D)
        else
            M1 = M0
        )
    ).

:- pred end_tiles(list(int)::in, array(int)::array_di,
    array(int)::array_uo) is det.
end_tiles([], !Owner).
end_tiles([P | Ps], !Owner) :-
    !Owner ^ elem(P) := 0,
    end_tiles(Ps, !Owner).

:- func ended(int) = int.
ended(P) = P << 8.

    % For each wanted position in turn, a tile there for its asker, unless
    % one has started there already.
:- pred want(list({int, int})::in, array(int)::array_di,
    array(int)::array_uo, array(int)::array_di, array(int)::array_uo,
    list(int)::in, list(int)::out) is det.
want([], !Owner, !Quiet, !New).
want([{There, Asker} | Wanted], !Owner, !Quiet, !New) :-
    ( if !.Owner ^ elem(There) = 0 then
        !Owner ^ elem(There) := Asker,
        !Quiet ^ elem(There) := 0,
        !:New = [(There << 8) \/ Asker | !.New]
    else
        true
    ),
    want(Wanted, !Owner, !Quiet, !New).

:- func changes(list(int)) = array(int32).
changes(Changes) = array(list.map(int32.cast_from_int, Changes)).

:- pred broadcast(list(mailbox(to_place))::in, to_place::in, io::di, io::uo)
    is det.
broadcast([], _, !IO).
broadcast([Place | Places], Message, !IO) :-
    send(Place, Message, !IO),
    broadcast(Places, Message, !IO).

    % The places' outcomes, summed.
:- pred outcomes(mailbox(to_directory)::in, int::in, int::in, int::out,
    int::in, int::out, int::in, int::out, io::di, io::uo) is det.
outcomes(Me, Left, !Population, !Checksum, !Trouble, !IO) :-
    ( if Left > 0 then
        receive(Me, any, after(600000), R, !IO),
        ( if R = message(outcome(_, Population, Checksum, Trouble)) then
            !:Population = !.Population + Population,
            !:Checksum = !.Checksum + Checksum,
            !:Trouble = int.max(!.Trouble, Trouble),
            outcomes(Me, Left - 1, !Population, !Checksum, !Trouble, !IO)
        else
            io.write_string("A place did not send its outcome.\n", !IO),
            exit_failure(!IO)
        )
    else
        true
    ).

    % The place in direction D of place P, if it is in the world.
:- pred beside(geometry::in, int::in, int::in, int::out) is semidet.
beside(G, P, D, Place) :-
    Side = G ^ side,
    Y = P / Side + D / 3 - 1,
    X = P mod Side + D mod 3 - 1,
    Y >= 0, Y < Side,
    X >= 0, X < Side,
    Place = Y * Side + X.

%---------------------------------------------------------------------------%
% A place.

    % Place Me of N: its peers, the tiles the directory gives it at the
    % start, their cells and halos, then Steps steps, and its outcome.
:- process place(mailbox(to_directory), int, int, int, int, int, int, int)
    receives to_place on gpu.

place(Directory, Me, N, World, Log, Soup, Steps, Capacity, Self, !IO) :-
    T = 1 << Log,
    Side = World >> Log,
    G = geometry(T, Log, Side, World),
    array.init(N, Self, Peers0),
    peers(Self, N, Peers0, Peers, !IO),
    send(Directory, ready(Me), !IO),
    array.init(Capacity * T * T, 0u8, CellsA0),
    array.init(Capacity * T * T, 0u8, CellsB),
    array.init(Side * Side, 0i32, Dir0),
    array.init(Side * Side, 0i32, Owner0),
    array.init(Capacity, 0i32, Pos0),
    array.init(Capacity * 9, 0i32, Nb0),
    array.init(Capacity, 0i32, Live0),
    array.init(Capacity, 0u16, Edges),
    array.init(Capacity, 0i32, Actives0),
    array.init(Capacity, 0i32, Ghosts0),
    array.init(Capacity, 0i32, Free0),
    % Tile 0 is the dead tile; the free list gives 1, 2 and so on first.
    ( for(J, 0, Capacity - 2), fromto(Free0, F0, F1, Free1) do
        F1 = F0 ^ unsafe_elem(J) := int32.cast_from_int(Capacity - 1 - J)
    ),
    From = (World - Soup) / 2,
    W = site(Me, Self, From, Soup),
    agree(G, W, Peers, -1, CellsA0, CellsA, Dir0, Dir1, Owner0, Owner1, Pos0, Pos1,
        Nb0, Nb1, Live0, Live1, Actives0, Actives1, 0, NActive1, Ghosts0,
        Ghosts1, 0, NGhost1, Free1, Free2, Capacity - 1, NFree1, 0, Trouble1,
        !IO),
    run(G, W, Peers, Directory, 0, Steps, CellsA, CellsB, Final, Dir1, _, Owner1, _,
        Pos1, Pos, Nb1, _, Live1, Live, Edges, _, Actives1, Actives, NActive1,
        NActive, Ghosts1, _, NGhost1, _, Free2, _, NFree1, _, Trouble1,
        Trouble, !IO),
    count(G, NActive, Actives, Pos, Live, Final, Population, Checksum),
    send(Directory, outcome(Me, Population, Checksum, Trouble), !IO).

    % What a place's agreement needs besides its tiles and its peers: its
    % number, its mailbox, and the soup's place and side.
:- type site
    --->    site(
                me :: int,
                self :: mailbox(to_place),
                from :: int,
                soup :: int
            ).

    % The N places' mailboxes, by their numbers.
:- pred peers(mailbox(to_place)::in, int::in,
    array(mailbox(to_place))::array_di, array(mailbox(to_place))::array_uo,
    io::di, io::uo) is det.
peers(Self, Left, !Peers, !IO) :-
    ( if Left > 0 then
        receive(Self, any, infinity, R, !IO),
        ( if R = message(peer(K, M)) then
            !Peers ^ elem(K) := M
        else
            true
        ),
        peers(Self, Left - 1, !Peers, !IO)
    else
        true
    ).

    % Steps from S to Steps, from Cur into Next, which swap after each; each
    % step's summary to the directory and the agreement after it. Final is
    % the cells after the last.
:- pred run(geometry::in, site::in, array(mailbox(to_place))::in,
    mailbox(to_directory)::in, int::in,
    int::in, array(uint8)::array_di, array(uint8)::array_di,
    array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint16)::array_di, array(uint16)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out, io::di, io::uo) is det.
run(G, W, Peers, Directory, S, Steps, Cur, Next0, Final, !Dir, !Owner, !Pos, !Nb,
        !Live, !Edges, !Actives, !NActive, !Ghosts, !NGhost, !Free, !NFree,
        !Trouble, !IO) :-
    ( if S >= Steps then
        Final = Cur
    else
        step(G, !.NActive, !.Actives, !.Nb, Cur, Next0, Next1, !Live,
            !Edges),
        Summary = summary(!.NActive, !.Actives, !.Pos, !.Live, !.Edges),
        send(Directory, summary(W ^ me, int32.cast_from_int(S), !.Trouble,
            Summary), !IO),
        agree(G, W, Peers, S, Next1, Next, !Dir, !Owner, !Pos, !Nb, !Live,
            !Actives, !NActive, !Ghosts, !NGhost, !Free, !NFree, !Trouble,
            !IO),
        run(G, W, Peers, Directory, S + 1, Steps, Next, Cur, Final, !Dir, !Owner,
            !Pos, !Nb, !Live, !Edges, !Actives, !NActive, !Ghosts, !NGhost,
            !Free, !NFree, !Trouble, !IO)
    ).

    % Each of the place's tiles, as the directory reads it.
:- func summary(int, array(int32), array(int32), array(int32),
    array(uint16)) = array(int32).
:- mode summary(in, array_ui, array_ui, array_ui, array_ui) = array_uo is det.
summary(NActive, Actives, Pos, Live, Edges) = Summary :-
    array.init(int.max(NActive, 1), 0i32, Summary0),
    ( for(J, 0, NActive - 1), fromto(Summary0, S0, S1, Summary1) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        Dead = ( if Live ^ unsafe_elem(I) = 0i32 then 1 else 0 ),
        S1 = S0 ^ unsafe_elem(J) := int32.cast_from_int(
            (int32.to_int(Pos ^ unsafe_elem(I)) << 10) \/ (Dead << 9) \/
            uint16.to_int(Edges ^ unsafe_elem(I)))
    ),
    % An array is at least one element long; a place with no tiles sends
    % none.
    ( if NActive = 0 then
        array.init(0, 0i32, Summary)
    else
        Summary = Summary1
    ).

    % The agreement after step S, or at the start for -1, into Cells: the
    % directory's map, which the place applies before it sends its halos,
    % and a halo from each place that owns a tile next to one of its own,
    % before or after the map.
:- pred agree(geometry::in, site::in, array(mailbox(to_place))::in,
    int::in,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out, io::di, io::uo) is det.
agree(G, W, Peers, S, !Cells, !Dir, !Owner, !Pos, !Nb, !Live, !Actives, !NActive,
        !Ghosts, !NGhost, !Free, !NFree, !Trouble, !IO) :-
    take(G, W, Peers, S, 0, 0, 0, !Cells, !Dir, !Owner, !Pos, !Nb, !Live, !Actives,
        !NActive, !Ghosts, !NGhost, !Free, !NFree, !Trouble, !IO).

    % Takes the agreement's messages until it has the map and a halo from
    % each place in Expected, which the map says. Got says which places'
    % halos have come.
:- pred take(geometry::in, site::in, array(mailbox(to_place))::in,
    int::in, int::in, int::in, int::in,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out, io::di, io::uo) is det.
take(G, W, Peers, S, GotMap, Got, Expected, !Cells, !Dir, !Owner, !Pos, !Nb, !Live,
        !Actives, !NActive, !Ghosts, !NGhost, !Free, !NFree, !Trouble,
        !IO) :-
    ( if GotMap = 1, Got /\ Expected = Expected then
        ( if Got = Expected then
            true
        else
            !:Trouble = 2
        )
    else
        receive(W ^ self, any, infinity, R, !IO),
        ( if
            R = message(map(Step, Changes)),
            int32.to_int(Step) = S,
            GotMap = 0
        then
            apply_map(G, W, S, Changes, !Cells, !Dir, !Owner, !Pos, !Nb,
                !Live, !Actives, !NActive, !Ghosts, !NGhost, !Free, !NFree,
                !Trouble),
            Expected1 = expected(G, W ^ me, !.NActive, !.Actives, !.Pos,
                !.Owner),
            send_halos(G, W, Peers, S, 0, Expected1, !.Cells, !.NActive, !.Actives,
                !.Pos, !.Owner, !IO),
            take(G, W, Peers, S, 1, Got, Expected1, !Cells, !Dir, !Owner, !Pos,
                !Nb, !Live, !Actives, !NActive, !Ghosts, !NGhost, !Free,
                !NFree, !Trouble, !IO)
        else if
            R = message(halo(From, Step, Halo)),
            int32.to_int(Step) = S,
            Got /\ (1 << From) = 0
        then
            apply_halo(G, Halo, !Cells, !Dir, !Pos, !Nb, !Ghosts, !NGhost,
                !Free, !NFree, !Trouble),
            take(G, W, Peers, S, GotMap, Got \/ (1 << From), Expected, !Cells,
                !Dir, !Owner, !Pos, !Nb, !Live, !Actives, !NActive, !Ghosts,
                !NGhost, !Free, !NFree, !Trouble, !IO)
        else
            % A message out of turn; the place goes on, and the directory
            % reports it at the end.
            !:Trouble = 2
        )
    ).

    % The directory's changes after step S: the owners in the place's copy
    % of the map, its tiles that end, the ghosts it no longer needs, which
    % are of tiles that end or are next to none of its own, and its new
    % tiles, all dead; at the start, its tiles' cells from the soup.
:- pred apply_map(geometry::in, site::in, int::in, array(int32)::in,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out) is det.
apply_map(G, W, S, Changes, !Cells, !Dir, !Owner, !Pos, !Nb, !Live,
        !Actives, !NActive, !Ghosts, !NGhost, !Free, !NFree, !Trouble) :-
    Mine = W ^ me + 1,
    NChanges = array.size(Changes),
    ( for(K, 0, NChanges - 1), fromto(!.Owner, O0, O1, !:Owner) do
        C = int32.to_int(Changes ^ unsafe_elem(K)),
        O1 = O0 ^ unsafe_elem(C >> 8) := int32.cast_from_int(C /\ 0xff)
    ),
    end_own(Mine, 0, !.Owner, !Dir, !.Pos, !Nb, !Live, !Actives, !NActive,
        !Free, !NFree),
    drop_ghosts(G, Mine, 0, !.Owner, !Dir, !.Pos, !Nb, !Ghosts, !NGhost,
        !Free, !NFree),
    Kept = !.NActive,
    ( for(K, 0, NChanges - 1), fromto(!.Dir, D0, D1, !:Dir),
        fromto(!.Pos, P0, P1, !:Pos), fromto(!.Nb, N0, N1, !:Nb),
        fromto(!.Live, L0, L1, !:Live), fromto(!.Actives, A0, A1, !:Actives),
        fromto(!.NActive, C0, C1, !:NActive), fromto(!.Free, F0, F1, !:Free),
        fromto(!.NFree, R0, R1, !:NFree),
        fromto(!.Trouble, T0, T1, !:Trouble) do
        C = int32.to_int(Changes ^ unsafe_elem(K)),
        ( if C /\ 0xff = Mine then
            start(G, C >> 8, I, D0, D1, P0, P1, N0, N1, A0, A1, C0, C1, F0,
                F1, R0, R1, T0, T1),
            L1 = L0 ^ unsafe_elem(I) := 0i32
        else
            D1 = D0, P1 = P0, N1 = N0, L1 = L0, A1 = A0, C1 = C0, F1 = F0,
            R1 = R0, T1 = T0
        )
    ),
    ( if S < 0 then
        init(G, W ^ from, W ^ soup, !.NActive, !.Actives, !.Pos, !Cells)
    else if !.NActive > Kept then
        % A loop across the teams costs microseconds even with nothing to
        % do.
        zero(G, Kept, !.NActive, !.Actives, !Cells)
    else
        true
    ).

    % From the J'th of the place's tiles on: each that the map no longer
    % says is its own ends, its place in the actives taken by the last one.
:- pred end_own(int::in, int::in, array(int32)::array_ui,
    array(int32)::array_di, array(int32)::array_uo, array(int32)::array_ui,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out) is det.
end_own(Mine, J, Owner, !Dir, Pos, !Nb, !Live, !Actives, !NActive, !Free,
        !NFree) :-
    ( if J >= !.NActive then
        true
    else
        I = int32.to_int(!.Actives ^ unsafe_elem(J)),
        P = int32.to_int(Pos ^ unsafe_elem(I)),
        ( if int32.to_int(Owner ^ unsafe_elem(P)) = Mine then
            end_own(Mine, J + 1, Owner, !Dir, Pos, !Nb, !Live, !Actives,
                !NActive, !Free, !NFree)
        else
            release(I, P, !Dir, !Nb, !Free, !NFree),
            !Live ^ unsafe_elem(I) := 0i32,
            !:NActive = !.NActive - 1,
            !Actives ^ unsafe_elem(J) := !.Actives ^ unsafe_elem(!.NActive),
            end_own(Mine, J, Owner, !Dir, Pos, !Nb, !Live, !Actives,
                !NActive, !Free, !NFree)
        )
    ).

    % From the J'th ghost on: each of a tile that has ended, or next to
    % none of the place's tiles, goes.
:- pred drop_ghosts(geometry::in, int::in, int::in, array(int32)::array_ui,
    array(int32)::array_di, array(int32)::array_uo, array(int32)::array_ui,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out) is det.
drop_ghosts(G, Mine, J, Owner, !Dir, Pos, !Nb, !Ghosts, !NGhost, !Free,
        !NFree) :-
    ( if J >= !.NGhost then
        true
    else
        I = int32.to_int(!.Ghosts ^ unsafe_elem(J)),
        P = int32.to_int(Pos ^ unsafe_elem(I)),
        Owned = Owner ^ unsafe_elem(P),
        Near = owns_beside(G, Owner, P, Mine),
        ( if Owned \= 0i32, Near = 1 then
            drop_ghosts(G, Mine, J + 1, Owner, !Dir, Pos, !Nb, !Ghosts,
                !NGhost, !Free, !NFree)
        else
            release(I, P, !Dir, !Nb, !Free, !NFree),
            !:NGhost = !.NGhost - 1,
            !Ghosts ^ unsafe_elem(J) := !.Ghosts ^ unsafe_elem(!.NGhost),
            drop_ghosts(G, Mine, J, Owner, !Dir, Pos, !Nb, !Ghosts, !NGhost,
                !Free, !NFree)
        )
    ).

    % Tile I, at position P, unlinked from its neighbours, out of the
    % directory, and back on the free list.
:- pred release(int::in, int::in,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out) is det.
release(I, P, !Dir, !Nb, !Free, !NFree) :-
    unlink(I, 0, !Nb),
    !Dir ^ unsafe_elem(P) := 0i32,
    !Free ^ unsafe_elem(!.NFree) := int32.cast_from_int(I),
    !:NFree = !.NFree + 1.

    % 1 if a tile next to position P is place Mine's, by the map.
:- func owns_beside(geometry, array(int32), int, int) = int.
:- mode owns_beside(in, array_ui, in, in) = out is det.
owns_beside(G, Owner, P, Mine) = Owns :-
    ( for(D, 0, 8), fromto(0, O0, O1, Owns) do
        ( if
            D \= 4,
            beside(G, P, D, There),
            int32.to_int(Owner ^ unsafe_elem(There)) = Mine
        then
            O1 = 1
        else
            O1 = O0
        )
    ).

    % The places, a bit each, that own a tile next to one of place Me's.
:- func expected(geometry, int, int, array(int32), array(int32),
    array(int32)) = int.
:- mode expected(in, in, in, array_ui, array_ui, array_ui) = out is det.
expected(G, Me, NActive, Actives, Pos, Owner) = Expected :-
    ( for(J, 0, NActive - 1), fromto(0, E0, E1, Expected) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        E1 = E0 \/ neighbours(G, Me, int32.to_int(Pos ^ unsafe_elem(I)),
            Owner)
    ).

    % The places other than Me, a bit each, that own a tile next to
    % position P.
:- func neighbours(geometry, int, int, array(int32)) = int.
:- mode neighbours(in, in, in, array_ui) = out is det.
neighbours(G, Me, P, Owner) = Places :-
    ( for(D, 0, 8), fromto(0, N0, N1, Places) do
        ( if
            D \= 4,
            beside(G, P, D, There),
            O = int32.to_int(Owner ^ unsafe_elem(There)),
            O \= 0,
            O \= Me + 1
        then
            N1 = N0 \/ (1 << (O - 1))
        else
            N1 = N0
        )
    ).

    % From place Q on, to each in Expected: the halo of the place's tiles
    % next to Q's.
:- pred send_halos(geometry::in, site::in, array(mailbox(to_place))::in,
    int::in, int::in, int::in, array(uint8)::array_ui, int::in,
    array(int32)::array_ui, array(int32)::array_ui, array(int32)::array_ui,
    io::di, io::uo) is det.
send_halos(G, W, Peers, S, Q, Expected, Cells, NActive, Actives, Pos, Owner,
        !IO) :-
    ( if Q >= array.size(Peers) then
        true
    else
        ( if Expected /\ (1 << Q) \= 0 then
            select(G, Q, NActive, Actives, Pos, Owner, K, Selected),
            Halo = pack(G, K, Selected, Pos, Cells),
            send(Peers ^ elem(Q), halo(W ^ me, int32.cast_from_int(S), Halo),
                !IO)
        else
            true
        ),
        send_halos(G, W, Peers, S, Q + 1, Expected, Cells, NActive, Actives,
            Pos, Owner, !IO)
    ).

    % The place's K tiles next to a tile of place Q.
:- pred select(geometry::in, int::in, int::in, array(int32)::array_ui,
    array(int32)::array_ui, array(int32)::array_ui, int::out,
    array(int32)::array_uo) is det.
select(G, Q, NActive, Actives, Pos, Owner, K, Selected) :-
    array.init(int.max(NActive, 1), 0i32, Selected0),
    ( for(J, 0, NActive - 1), fromto(Selected0, S0, S1, Selected),
        fromto(0, K0, K1, K) do
        I = Actives ^ unsafe_elem(J),
        ( if
            owns_beside(G, Owner,
                int32.to_int(Pos ^ unsafe_elem(int32.to_int(I))), Q + 1) = 1
        then
            S1 = S0 ^ unsafe_elem(K0) := I,
            K1 = K0 + 1
        else
            S1 = S0,
            K1 = K0
        )
    ).

    % The halo of the K selected tiles, across the teams: each one's
    % position, then its border cells.
:- func pack(geometry, int, array(int32), array(int32), array(uint8)) =
    array(uint8).
:- mode pack(in, in, array_ui, array_ui, array_ui) = array_uo is det.
pack(G, K, Selected, Pos, Cells) = Halo :-
    Bytes = 4 * G ^ tile,
    array.init(K * Bytes, 0u8, Halo0),
    %$omp distribute
    ( for(J, 0, K - 1), fromto(Halo0, H0, H1, Halo) do
        I = int32.to_int(Selected ^ unsafe_elem(J)),
        P = int32.to_int(Pos ^ unsafe_elem(I)),
        Base = J * Bytes,
        H2 = H0 ^ unsafe_elem(Base) := uint8.cast_from_int(P /\ 0xff),
        H3 = H2 ^ unsafe_elem(Base + 1) :=
            uint8.cast_from_int((P >> 8) /\ 0xff),
        H4 = H3 ^ unsafe_elem(Base + 2) :=
            uint8.cast_from_int((P >> 16) /\ 0xff),
        H5 = H4 ^ unsafe_elem(Base + 3) :=
            uint8.cast_from_int((P >> 24) /\ 0xff),
        pack_tile(G, I, Cells, Base + 4, H5, H1)
    ).

:- pred pack_tile(geometry::in, int::in, array(uint8)::array_ui, int::in,
    array(uint8)::array_di, array(uint8)::array_uo) is det.
pack_tile(G, I, Cells, Base, !Halo) :-
    T = G ^ tile,
    Log = G ^ log,
    %$omp parallel do
    ( for(R, 0, 4 * T - 5) do
        !Halo ^ unsafe_elem(Base + R) :=
            Cells ^ unsafe_elem((I << (2 * Log)) + ring(Log, T, R))
    ).

    % The index in its tile of a tile's R'th border cell: its first row, its
    % last, then its first column and its last, without their corners.
:- func ring(int, int, int) = int.
ring(Log, T, R) =
    ( if R < T then
        R
    else if R < 2 * T then
        ((T - 1) << Log) + R - T
    else if R < 3 * T - 2 then
        (R - 2 * T + 1) << Log
    else
        ((R - 3 * T + 3) << Log) + T - 1
    ).

    % A halo's cells into the ghosts of its tiles, each started if the place
    % has none at its position.
:- pred apply_halo(geometry::in, array(uint8)::in,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out) is det.
apply_halo(G, Halo, !Cells, !Dir, !Pos, !Nb, !Ghosts, !NGhost, !Free,
        !NFree, !Trouble) :-
    Bytes = 4 * G ^ tile,
    K = array.size(Halo) / Bytes,
    array.init(int.max(K, 1), 0i32, Slots0),
    ( for(J, 0, K - 1), fromto(Slots0, S0, S1, Slots),
        fromto(!.Dir, D0, D1, !:Dir), fromto(!.Pos, P0, P1, !:Pos),
        fromto(!.Nb, N0, N1, !:Nb), fromto(!.Ghosts, G0, G1, !:Ghosts),
        fromto(!.NGhost, C0, C1, !:NGhost), fromto(!.Free, F0, F1, !:Free),
        fromto(!.NFree, R0, R1, !:NFree),
        fromto(!.Trouble, T0, T1, !:Trouble) do
        Base = J * Bytes,
        P = uint8.to_int(Halo ^ unsafe_elem(Base)) \/
            (uint8.to_int(Halo ^ unsafe_elem(Base + 1)) << 8) \/
            (uint8.to_int(Halo ^ unsafe_elem(Base + 2)) << 16) \/
            (uint8.to_int(Halo ^ unsafe_elem(Base + 3)) << 24),
        Found = int32.to_int(D0 ^ unsafe_elem(P)),
        ( if Found \= 0 then
            I = Found,
            D1 = D0, P1 = P0, N1 = N0, G1 = G0, C1 = C0, F1 = F0, R1 = R0,
            T1 = T0
        else
            start(G, P, I, D0, D1, P0, P1, N0, N1, G0, G1, C0, C1, F0, F1, R0,
                R1, T0, T1)
        ),
        S1 = S0 ^ unsafe_elem(J) := int32.cast_from_int(I)
    ),
    copy(G, K, Slots, Halo, !Cells).

    % The K tiles' border cells from the halo into their ghosts, across the
    % teams; a ghost that the pool had no room for is left out.
:- pred copy(geometry::in, int::in, array(int32)::array_ui,
    array(uint8)::in, array(uint8)::array_di, array(uint8)::array_uo) is det.
copy(G, K, Slots, Halo, !Cells) :-
    Bytes = 4 * G ^ tile,
    %$omp distribute
    ( for(J, 0, K - 1) do
        I = int32.to_int(Slots ^ unsafe_elem(J)),
        ( if I \= 0 then
            copy_tile(G, I, Halo, J * Bytes + 4, !Cells)
        else
            true
        )
    ).

:- pred copy_tile(geometry::in, int::in, array(uint8)::in, int::in,
    array(uint8)::array_di, array(uint8)::array_uo) is det.
copy_tile(G, I, Halo, Base, !Cells) :-
    T = G ^ tile,
    Log = G ^ log,
    %$omp parallel do
    ( for(R, 0, 4 * T - 5) do
        !Cells ^ unsafe_elem((I << (2 * Log)) + ring(Log, T, R)) :=
            Halo ^ unsafe_elem(Base + R)
    ).

    % A tile at position Place, I, from the free list, linked to its
    % neighbours both ways and on the list List; or I = 0, the dead tile, and
    % Trouble 1 if there is none free. A free tile's live cells are 0, as a
    % tile ends only with none.
:- pred start(geometry::in, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out) is det.
start(G, Place, I, !Dir, !Pos, !Nb, !List, !NList, !Free, !NFree,
        !Trouble) :-
    ( if !.NFree = 0 then
        I = 0,
        !:Trouble = int.max(!.Trouble, 1)
    else
        !:NFree = !.NFree - 1,
        I = int32.to_int(!.Free ^ unsafe_elem(!.NFree)),
        !Dir ^ unsafe_elem(Place) := int32.cast_from_int(I),
        !Pos ^ unsafe_elem(I) := int32.cast_from_int(Place),
        link(G, I, Place, 0, !.Dir, !Nb),
        !List ^ unsafe_elem(!.NList) := int32.cast_from_int(I),
        !:NList = !.NList + 1
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
% Links between tiles, and loops across the teams: the soup's cells, new
% tiles' zeros, and the count.

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

% Ends the program at once with status 1, when a place stops answering and
% the protocol cannot go on: the other places may still run.
:- pred exit_failure(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    exit_failure(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    fflush(stdout);
    exit(1);
").


:- pred header(int::in, int::in, int::in, int::in, int::in, int::in,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    header(World::in, Tile::in, Capacity::in, Soup::in, Steps::in,
        Places::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Game of Life, Mercury, GPU processes, sparse tiles, %ld \"
        \"places\\n\", (long) Places);
    printf(\"World: %ld x %ld cells, tiles of %ld x %ld, %ld in each \"
        \"place's pool\\n\", (long) World, (long) World, (long) Tile,
        (long) Tile, (long) Capacity - 1);
    printf(\"Soup: %ld x %ld cells at the centre\\n\", (long) Soup,
        (long) Soup);
    printf(\"Steps: %ld\\n\", (long) Steps);
").

    % The time of each step but the first, as min, max and average, as the
    % directory sees it, from one map to the next; and the active tiles.
:- pred results(int::in, array(float)::in, int::in, int::in, int::in,
    int::in, io::di, io::uo) is det.
results(Steps, Times, Population, Checksum, Peak, Final, !IO) :-
    ( for(S, 1, Steps - 1),
        fromto(Times ^ elem(1), Min0, Min1, Min),
        fromto(Times ^ elem(1), Max0, Max1, Max),
        fromto(0.0, Sum0, Sum1, Sum) do
        Min1 = float.min(Min0, Times ^ elem(S)),
        Max1 = float.max(Max0, Times ^ elem(S)),
        Sum1 = Sum0 + Times ^ elem(S)
    ),
    write_results(Population, Checksum, Peak, Final, Min, Max,
        Sum / float(Steps - 1), !IO).

:- pred write_results(int::in, int::in, int::in, int::in, float::in,
    float::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Population::in, Checksum::in, Peak::in, Final::in, Min::in,
        Max::in, Average::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Population: %ld\\n\", (long) Population);
    printf(\"Checksum: %ld\\n\", (long) Checksum);
    printf(\"Active tiles: %ld at most, %ld at the end\\n\", (long) Peak,
        (long) Final);
    printf(\"%-12s%-12s%-12s\\n\", \"Min (sec)\", \"Max\", \"Average\");
    printf(\"%-12.6f%-12.6f%-12.6f\\n\", Min, Max, Average);
").
