% The shallow water equations with wetting and drying on sparse tiles, in a
% GPU process: only the tiles where there is water are stepped, starting in
% front of it and ending behind it, as life-sparse.m's tiles follow life
% (docs/INPROGRESS.md, A showcase with more work per tile). swe-sparse.c and
% swe-sparse.hip are the same program in C and HIP, and give the same bits.
%
% - The world is SWE_WORLD cells square, of side 1 m, in tiles of SWE_TILE
%   cells square, a power of two that divides it. Beyond its edge, and in a
%   tile that is not active, cells are dry.
% - The pool: two arrays of SWE_CAPACITY tiles' cells, the current step's
%   and the next, which swap after each step. Tile I's cells are 3 T^2
%   floats from I * 3 * T^2 on: its depths h by rows, then its momenta hu,
%   then hv. Tile 0 is the dry tile, all zero, never stepped, the neighbour
%   of a tile where there is none.
% - The directory, the tiles' fields, the actives and the free list are
%   life-sparse.m's, with a tile's wet cells for its live ones, its wet
%   edges for its live ones, and its largest wave speed.
% - The scheme: at each face of a cell, an HLL flux of the states either
%   side, with hydrostatic reconstruction over the bed
%   b = D0 (r^2 / R^2 - 1), a bowl at the world's centre, computed wherever
%   needed; explicit Euler in time, each step's dt 0.45 over the largest
%   |u| + |v| + 2 sqrt(g h) after the last. A cell at most 1e-6 m deep is
%   dry: its momenta are zero.
% - SWE_CASE 0: a column of water collapses and sloshes across the bowl;
%   1: Thacker's oscillating paraboloid, whose exact solution has a moving
%   shoreline, and the program prints the relative L1 error of the depth.
%
% A step is a loop under distribute over the active tiles, a team for each,
% with a loop under parallel do over its cells across the team's threads,
% which reduces its wet cells, its wet edges' bits and its largest speed.
% After it, the bookkeeping, as life-sparse.m's, which also finds the next
% step's dt. A wet cell on a tile's edge or corner starts the tile beyond
% it; a tile with no wet cell and no wet edge facing it for quiet_steps
% steps ends.
%
% Every operation is in the order of swe-sparse.c's, and merkc fuses no
% multiply and add, so the programs give the same bits: the checksum is the
% sum, modulo 2^64, over the active cells of
% (bits(h) + 3 bits(hu) + 5 bits(hv)) (2 g + 1), g the cell's index in the
% world and bits a float's, -0.0 as 0.0, which no order changes.
%
% The process sends the host each step's time, the bookkeeping's part of
% it, the active tiles after each step, and the outcome. SWE_TRACE=1 prints
% each step's line.
%
% Built with -fprocess-declarations, -fdo-loops and --offload-arch, and
% -fopenmp for the loops to run across the teams; without -fopenmp the
% process runs them on its own lane. The first argument `host` spawns the
% process at the host, for tests. See README.md.

:- module swe_sparse.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int32, list, math, process, string,
    uint16, uint8.

    % The wet cells, the checksum, the most active tiles at once, whether
    % the pool ran out of tiles, the simulated time, the volume, and the
    % error's sum and the exact depths' sum.
:- type outcome
    --->    outcome(int, int, int, int, float, float, float, float).

    % The world's shape: its tiles' side and its log, the world's side in
    % tiles, and in cells; and the case.
:- type geometry
    --->    geometry(
                tile :: int,
                log :: int,
                side :: int,
                world :: int,
                case :: int
            ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    setting("SWE_WORLD", 2048, World, !IO),
    setting("SWE_TILE", 32, Tile, !IO),
    setting("SWE_STEPS", 1000, Steps, !IO),
    setting("SWE_CASE", 0, Case, !IO),
    Log = log2(Tile, 0),
    Side = ( if Tile > 0 then World / Tile else 0 ),
    setting("SWE_CAPACITY", int.min(Side * Side, 4096) + 1, Capacity, !IO),
    setting("SWE_TRACE", 0, Trace, !IO),
    ( if
        Tile >= 4, 1 << Log = Tile,
        World >= Tile, World mod Tile = 0,
        Steps >= 2,
        Capacity >= 2,
        Case >= 0, Case =< 1
    then
        header(World, Tile, Capacity, Case, Steps, !IO),
        new_mailbox(TimesBox, !IO),
        new_mailbox(CountsBox, !IO),
        new_mailbox(OutcomeBox, !IO),
        spawn(Place, swe(TimesBox, CountsBox, OutcomeBox, World, Log, Case,
            Steps, Capacity), _, !IO),
        receive(TimesBox, any, after(600000), R1, !IO),
        receive(CountsBox, any, after(600000), R2, !IO),
        receive(OutcomeBox, any, after(600000), R3, !IO),
        ( if
            R1 = message(Times),
            R2 = message(Counts),
            R3 = message(outcome(Wet, Checksum, Peak, Overflow, Time, Volume,
                Error, Exact))
        then
            ( if Overflow = 0 then
                ( if Trace = 0 then
                    true
                else
                    trace(Steps, Times, Counts, !IO)
                ),
                results(Steps, Times, Counts, Case, Time, Wet, Volume,
                    Checksum, Error, Exact, Peak, !IO)
            else
                io.write_string("The pool ran out of tiles: " ++
                    "raise SWE_CAPACITY.\n", !IO),
                exit_failure(!IO)
            )
        else
            io.write_string("The GPU process did not answer.\n", !IO),
            exit_failure(!IO)
        )
    else
        io.write_string("SWE_TILE must be a power of two of at least 4 " ++
            "that divides SWE_WORLD, SWE_STEPS at least 2, SWE_CAPACITY " ++
            "at least 2, and SWE_CASE 0 or 1.\n", !IO),
        exit_failure(!IO)
    ).

    % The log of N to base 2, if N is a power of two.
:- func log2(int, int) = int.
log2(N, L) = ( if N =< 1 then L else log2(N >> 1, L + 1) ).

%---------------------------------------------------------------------------%
% The scheme, as swe.h has it for the C and HIP programs.

    % Gravity, the depth of a dry cell, the bowl's depth at its centre, the
    % column's depth above the bed, the Courant number, Thacker's amplitude.
:- func g = float.
g = 9.81.
:- func dry = float.
dry = 1.0e-6.
:- func d0 = float.
d0 = 10.0.
:- func column = float.
column = 5.0.
:- func courant = float.
courant = 0.45.
:- func a = float.
a = 0.4.

    % The bed at the centre of the cell at row Y and column X of a world of
    % side N, a bowl of radius 0.45 N.
    %
:- func bed(int, int, int) = float.
bed(Y, X, N) = d0 * (unchecked_quotient(DX * DX + DY * DY, R * R) - 1.0) :-
    C = 0.5 * float(N),
    R = 0.45 * float(N),
    DX = (float(X) + 0.5) - C,
    DY = (float(Y) + 0.5) - C.

    % Thacker's depth at the cell's centre at time T.
    %
:- func thacker(int, int, int, float) = float.
thacker(Y, X, N, T) = float.max(0.0, S - bed(Y, X, N)) :-
    C = 0.5 * float(N),
    R = 0.45 * float(N),
    DX = (float(X) + 0.5) - C,
    DY = (float(Y) + 0.5) - C,
    W = unchecked_quotient(unchecked_sqrt(8.0 * g * d0), R),
    Q = 1.0 - a * math.cos(W * T),
    S = d0 * (unchecked_quotient(unchecked_sqrt(1.0 - a * a), Q) - 1.0 -
        unchecked_quotient(DX * DX + DY * DY, R * R) *
        (unchecked_quotient(1.0 - a * a, Q * Q) - 1.0)).

    % The cell's depth at the start.
    %
:- func start_depth(int, int, int, int) = float.
start_depth(Y, X, N, Case) = H :-
    ( if Case = 1 then
        H = thacker(Y, X, N, 0.0)
    else
        C = 0.5 * float(N),
        CX = C + 0.2 * float(N),
        RC = unchecked_quotient(float(N), 16.0),
        DX = (float(X) + 0.5) - CX,
        DY = (float(Y) + 0.5) - C,
        ( if DX * DX + DY * DY > RC * RC then
            H = 0.0
        else
            % A flat top `column` above the bed under the column's centre.
            Top = d0 * (unchecked_quotient((0.2 * float(N)) * (0.2 * float(N)),
                (0.45 * float(N)) * (0.45 * float(N))) - 1.0) + column,
            H = float.max(0.0, Top - bed(Y, X, N))
        )
    ).

    % The flux through the face between a left and a right cell, each its
    % depth, its momenta normal to the face and along it, and its bed: the
    % mass, normal and along fluxes, and the left's and the right's pressure
    % corrections of hydrostatic reconstruction.
    %
:- pred hll(float::in, float::in, float::in, float::in, float::in, float::in,
    float::in, float::in, float::out, float::out, float::out, float::out,
    float::out) is det.
hll(HL, NL, AL, BL, HR, NR, AR, BR, FM, FN, FA, PL, PR) :-
    UL = ( if HL > dry then unchecked_quotient(NL, HL) else 0.0 ),
    VL = ( if HL > dry then unchecked_quotient(AL, HL) else 0.0 ),
    UR = ( if HR > dry then unchecked_quotient(NR, HR) else 0.0 ),
    VR = ( if HR > dry then unchecked_quotient(AR, HR) else 0.0 ),
    BF = float.max(BL, BR),
    HLS = float.max(0.0, HL + BL - BF),
    HRS = float.max(0.0, HR + BR - BF),
    CL = unchecked_sqrt(g * HLS),
    CR = unchecked_sqrt(g * HRS),
    SL = float.min(UL - CL, UR - CR),
    SR = float.max(UL + CL, UR + CR),
    FML = HLS * UL,
    FNL = HLS * UL * UL + 0.5 * g * HLS * HLS,
    FAL = HLS * UL * VL,
    FMR = HRS * UR,
    FNR = HRS * UR * UR + 0.5 * g * HRS * HRS,
    FAR = HRS * UR * VR,
    ( if SL >= 0.0 then
        FM = FML,
        FN = FNL,
        FA = FAL
    else if SR =< 0.0 then
        FM = FMR,
        FN = FNR,
        FA = FAR
    else
        D = SR - SL,
        FM = unchecked_quotient(SR * FML - SL * FMR + SL * SR * (HRS - HLS),
            D),
        FN = unchecked_quotient(SR * FNL - SL * FNR +
            SL * SR * (HRS * UR - HLS * UL), D),
        FA = unchecked_quotient(SR * FAL - SL * FAR +
            SL * SR * (HRS * VR - HLS * VL), D)
    ),
    PL = 0.5 * g * (HL * HL - HLS * HLS),
    PR = 0.5 * g * (HR * HR - HRS * HRS).

    % The cell's wave speed, |u| + |v| + 2 sqrt(g h), 0 if it is dry.
    %
:- func speed(float, float, float) = float.
speed(H, HU, HV) = S :-
    ( if H =< dry then
        S = 0.0
    else
        C = unchecked_sqrt(g * H),
        S = float.abs(unchecked_quotient(HU, H)) +
            float.abs(unchecked_quotient(HV, H)) + 2.0 * C
    ).

%---------------------------------------------------------------------------%
% The process.

    % Steps the world Steps times and sends the host the times, the active
    % tiles after each step, and the outcome.
:- process swe(mailbox(array(float)), mailbox(array(int)), mailbox(outcome),
    int, int, int, int, int) receives int on gpu.

swe(TimesBox, CountsBox, OutcomeBox, World, Log, Case, Steps, Capacity,
        _Self, !IO) :-
    T = 1 << Log,
    Side = World >> Log,
    G = geometry(T, Log, Side, World, Case),
    array.init(Capacity * 3 * T * T, 0.0, Cur0),
    array.init(Capacity * 3 * T * T, 0.0, Next),
    array.init(Side * Side, 0i32, Dir0),
    array.init(Capacity, 0i32, Pos0),
    array.init(Capacity * 9, 0i32, Nb0),
    array.init(Capacity, 0i32, Wet),
    array.init(Capacity, 0u16, Edges0),
    array.init(Capacity, 0.0, Speed0),
    array.init(Capacity, 0u8, Quiet0),
    array.init(Capacity, 0i32, Actives0),
    array.init(Capacity, 0i32, Free0),
    array.init(Capacity, 0u16, Fate0),
    % Tile 0 is the dry tile; the free list gives 1, 2 and so on first.
    ( for(J, 0, Capacity - 2), fromto(Free0, F0, F1, Free1) do
        F1 = F0 ^ unsafe_elem(J) := int32.cast_from_int(Capacity - 1 - J)
    ),
    % The tiles over the wet region's bounding square, and each tile's
    % edges all wet, so that the tiles around them start too.
    Half = ( if Case = 0 then unchecked_quotient(float(World), 16.0)
        else 0.45 * float(World) ),
    CX = 0.5 * float(World) + ( if Case = 0 then 0.2 * float(World)
        else 0.0 ),
    CY = 0.5 * float(World),
    X0 = int.max(0, float.floor_to_int(CX - Half)),
    X1 = int.min(World - 1, float.ceiling_to_int(CX + Half)),
    Y0 = int.max(0, float.floor_to_int(CY - Half)),
    Y1 = int.min(World - 1, float.ceiling_to_int(CY + Half)),
    first_tiles(G, Y0 >> Log, Y1 >> Log, X0 >> Log, X1 >> Log, Dir0, Dir1,
        Pos0, Pos1, Nb0, Nb1, Quiet0, Quiet1, Actives0, Actives1, 0, NActive1,
        Free1, Free2, Capacity - 1, NFree1, 0, Overflow1),
    init(G, NActive1, Actives1, Pos1, Cur0, Cur1, Speed0, Speed1),
    Fastest = fastest(NActive1, Actives1, Speed1),
    ( for(J, 0, NActive1 - 1), fromto(Fate0, E0, E1, Fate1) do
        E1 = E0 ^ unsafe_elem(J) := all_edges
    ),
    grow(G, 0, NActive1, Fate1, Fate, Dir1, Dir, Pos1, Pos, Nb1, Nb,
        Quiet1, Quiet, Actives1, Actives, NActive1, NActive, Free2, Free,
        NFree1, NFree, Overflow1, Overflow),
    array.init(2 * Steps, 0.0, Times0),
    array.init(Steps, 0, Counts0),
    run(G, 0, Steps, Fastest, 0.0, Time, Cur1, Next, Final, Dir, _, Pos,
        Pos2, Nb, _, Wet, WetF, Edges0, _, Speed1, _, Quiet, _, Fate, _,
        Actives, ActivesF, NActive, NActiveF, Free, _, NFree, _, Overflow,
        OverflowF, NActive, Peak, Times0, Times, Counts0, Counts, !IO),
    count(G, Time, NActiveF, ActivesF, Pos2, WetF, Final, WetCells, Checksum,
        Volume, Error, Exact),
    send(TimesBox, Times, !IO),
    send(CountsBox, Counts, !IO),
    send(OutcomeBox, outcome(WetCells, Checksum, Peak, OverflowF, Time,
        Volume, Error, Exact), !IO).

    % Every direction's bit but the middle's.
:- func all_edges = uint16.
all_edges = 0x1ef_u16.

    % The bit of a tile's fate that ends it, above the directions'.
:- func ends = int.
ends = 0x200.

    % The steps a tile with no water near it waits before it ends, so that
    % one that the water is about to reach again stays.
:- func quiet_steps = int.
quiet_steps = 4.

    % Steps from S to Steps, from Cur into Next, which swap after each, each
    % step's bookkeeping after it; Fastest is the largest speed before the
    % step, and Time the time simulated. Final is the cells after the last.
    %
:- pred run(geometry::in, int::in, int::in, float::in, float::in, float::out,
    array(float)::array_di, array(float)::array_di, array(float)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint16)::array_di, array(uint16)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(uint16)::array_di, array(uint16)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out, int::in, int::out,
    array(float)::array_di, array(float)::array_uo,
    array(int)::array_di, array(int)::array_uo, io::di, io::uo) is det.
run(G, S, Steps, Fastest, !Time, Cur, Next0, Final, !Dir, !Pos, !Nb, !Wet,
        !Edges, !Speed, !Quiet, !Fate, !Actives, !NActive, !Free, !NFree,
        !Overflow, !Peak, !Times, !Counts, !IO) :-
    ( if S >= Steps then
        Final = Cur
    else if !.Overflow \= 0 then
        Final = Cur
    else
        now(T0, !IO),
        DT = unchecked_quotient(courant, float.max(Fastest, dry)),
        step(G, DT, !.NActive, !.Actives, !.Pos, !.Nb, Cur, Next0, Next,
            !Wet, !Edges, !Speed),
        now(T1, !IO),
        !:Time = !.Time + DT,
        decide(G, !.NActive, !.Actives, !.Pos, !.Nb, !.Wet, !.Edges, !.Speed,
            !Quiet, !Fate, Pending, Fastest1),
        ( if Pending = 0 then
            Next1 = Next
        else
            finish(0, !Fate, !Dir, !.Pos, !Nb, !Actives, !NActive, !Free,
                !NFree),
            Kept = !.NActive,
            grow(G, 0, Kept, !Fate, !Dir, !Pos, !Nb, !Quiet, !Actives,
                !NActive, !Free, !NFree, !Overflow),
            % A loop across the teams costs microseconds even with nothing
            % to do.
            ( if !.NActive > Kept then
                zero(G, Kept, !.NActive, !.Actives, Next, Next1)
            else
                Next1 = Next
            )
        ),
        now(T2, !IO),
        !Times ^ elem(S) := T2 - T0,
        !Times ^ elem(Steps + S) := T2 - T1,
        !Counts ^ elem(S) := !.NActive,
        !:Peak = int.max(!.Peak, !.NActive),
        run(G, S + 1, Steps, Fastest1, !Time, Next1, Cur, Final, !Dir, !Pos,
            !Nb, !Wet, !Edges, !Speed, !Quiet, !Fate, !Actives, !NActive,
            !Free, !NFree, !Overflow, !Peak, !Times, !Counts, !IO)
    ).

%---------------------------------------------------------------------------%
% The step.

    % Next's cells of the active tiles from Cur's, a step of DT, and each
    % tile's wet cells, wet edges and largest speed: a team for each tile.
    %
:- pred step(geometry::in, float::in, int::in, array(int32)::array_ui,
    array(int32)::array_ui, array(int32)::array_ui, array(float)::array_ui,
    array(float)::array_di, array(float)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint16)::array_di, array(uint16)::array_uo,
    array(float)::array_di, array(float)::array_uo) is det.
step(G, DT, NActive, Actives, Pos, Nb, Cur, !Next, !Wet, !Edges, !Speed) :-
    %$omp distribute
    ( for(J, 0, NActive - 1) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        P = int32.to_int(Pos ^ unsafe_elem(I)),
        step_tile(G, DT, I, P, Nb, Cur, !Next, Wet, Edges, Fastest),
        !Wet ^ unsafe_elem(I) := int32.cast_from_int(Wet),
        !Edges ^ unsafe_elem(I) := uint16.cast_from_int(Edges),
        !Speed ^ unsafe_elem(I) := Fastest
    ).

    % Tile I's cells, at place P, across the team's threads.
    %
:- pred step_tile(geometry::in, float::in, int::in, int::in,
    array(int32)::array_ui, array(float)::array_ui,
    array(float)::array_di, array(float)::array_uo,
    int::out, int::out, float::out) is det.
step_tile(G, DT, I, P, Nb, Cur, !Next, Wet, Edges, Fastest) :-
    T = G ^ tile,
    Log = G ^ log,
    N = G ^ world,
    M = T - 1,
    TT = T * T,
    Base = I * 9,
    Cells = I * 3 * TT,
    Top = (P / G ^ side) << Log,
    Left = (P mod G ^ side) << Log,
    %$omp parallel do reduction((+) : Wet) reduction(\/ : Edges)
    %$omp& reduction(max : Fastest)
    ( for(K, 0, TT - 1), fromto(0, W0, W1, Wet), fromto(0, E0, E1, Edges),
        fromto(0.0, S0, S1, Fastest) do
        R = K >> Log,
        C = K /\ M,
        Y = Top + R,
        X = Left + C,
        cell(Log, M, TT, Base, Nb, Cur, R, C, H, HU, HV),
        cell(Log, M, TT, Base, Nb, Cur, R, C - 1, HW, HUW, HVW),
        cell(Log, M, TT, Base, Nb, Cur, R, C + 1, HE, HUE, HVE),
        cell(Log, M, TT, Base, Nb, Cur, R - 1, C, HN, HUN, HVN),
        cell(Log, M, TT, Base, Nb, Cur, R + 1, C, HS, HUS, HVS),
        B = bed(Y, X, N),
        BW = bed(Y, X - 1, N),
        BE = bed(Y, X + 1, N),
        BN = bed(Y - 1, X, N),
        BS = bed(Y + 1, X, N),
        % Across x the normal momentum is hu; across y, hv.
        hll(HW, HUW, HVW, BW, H, HU, HV, B, FWM, FWN, FWA, _, FWPR),
        hll(H, HU, HV, B, HE, HUE, HVE, BE, FEM, FEN, FEA, FEPL, _),
        hll(HN, HVN, HUN, BN, H, HV, HU, B, FNM, FNN, FNA, _, FNPR),
        hll(H, HV, HU, B, HS, HVS, HUS, BS, FSM, FSN, FSA, FSPL, _),
        NH0 = H - DT * ((FEM - FWM) + (FSM - FNM)),
        NHU0 = HU - DT * (((FEN + FEPL) - (FWN + FWPR)) + (FSA - FNA)),
        NHV0 = HV - DT * ((FEA - FWA) + ((FSN + FSPL) - (FNN + FNPR))),
        NH = float.max(NH0, 0.0),
        ( if NH =< dry then
            NHU = 0.0,
            NHV = 0.0
        else
            NHU = NHU0,
            NHV = NHV0
        ),
        Next1 = !.Next ^ unsafe_elem(Cells + K) := NH,
        Next2 = Next1 ^ unsafe_elem(Cells + TT + K) := NHU,
        !:Next = Next2 ^ unsafe_elem(Cells + 2 * TT + K) := NHV,
        ( if NH > dry then
            W1 = W0 + 1,
            E1 = E0 \/ edge_bits(R, C, M)
        else
            W1 = W0,
            E1 = E0
        ),
        S1 = float.max(S0, speed(NH, NHU, NHV))
    ).

    % The state of the cell at row R and column C of the tile whose
    % neighbours start at Base, each from -1 to the tile's side: in the
    % tile, or in the neighbour that R and C reach, at the row and column
    % there.
    %
:- pred cell(int::in, int::in, int::in, int::in, array(int32)::array_ui,
    array(float)::array_ui, int::in, int::in, float::out, float::out,
    float::out) is det.
cell(Log, M, TT, Base, Nb, Cur, R, C, H, HU, HV) :-
    D = ((R >> Log) + 1) * 3 + (C >> Log) + 1,
    Tile = int32.to_int(Nb ^ unsafe_elem(Base + D)),
    At = Tile * 3 * TT + ((R /\ M) << Log) + (C /\ M),
    H = Cur ^ unsafe_elem(At),
    HU = Cur ^ unsafe_elem(At + TT),
    HV = Cur ^ unsafe_elem(At + 2 * TT).

    % The directions that a wet cell at row R and column C makes wet: the
    % edge it is on, if any, each, and the corner.
    %
:- func edge_bits(int, int, int) = int.
edge_bits(R, C, M) = bit(Row, 0) \/ bit(0, Column) \/ bit(Row, Column) :-
    Row = ( if R = 0 then -1 else if R = M then 1 else 0 ),
    Column = ( if C = 0 then -1 else if C = M then 1 else 0 ).

:- func bit(int, int) = int.
bit(DY, DX) = ( if DY = 0, DX = 0 then 0 else 1 << ((DY + 1) * 3 + DX + 1) ).

%---------------------------------------------------------------------------%
% The bookkeeping, life-sparse.m's.

    % Each active tile's fate after a step, from the step's results alone,
    % across the process's teams: its quiet steps, which end it at
    % quiet_steps, and the directions where it starts a tile, each a bit of
    % its element of Fate, which is by its place in the actives; how many
    % tiles have a fate; and the largest speed, for the next step. A tile's
    % wet cells and edges, and its neighbours' edges, are the step's; the
    % dry tile's edges are none.
    %
:- pred decide(geometry::in, int::in, array(int32)::array_ui,
    array(int32)::array_ui, array(int32)::array_ui, array(int32)::array_ui,
    array(uint16)::array_ui, array(float)::array_ui,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(uint16)::array_di, array(uint16)::array_uo, int::out, float::out)
    is det.
decide(G, NActive, Actives, Pos, Nb, Wet, Edges, Speed, !Quiet, !Fate,
        Pending, Fastest) :-
    %$omp distribute parallel do reduction((+) : Pending)
    %$omp& reduction(max : Fastest)
    ( for(J, 0, NActive - 1), fromto(0, P0, P1, Pending),
        fromto(0.0, S0, S1, Fastest) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        Base = I * 9,
        Place = int32.to_int(Pos ^ unsafe_elem(I)),
        Starts = uint16.to_int(Edges ^ unsafe_elem(I)) /\
            missing(Nb, Base) /\ inside(G, Place),
        ( if
            Wet ^ unsafe_elem(I) = 0i32,
            facing(Nb, Edges, Base) = 0
        then
            Q = uint8.to_int(!.Quiet ^ unsafe_elem(I)) + 1
        else
            Q = 0
        ),
        !Quiet ^ unsafe_elem(I) := uint8.cast_from_int(Q),
        F = ( if Q >= quiet_steps then Starts \/ ends else Starts ),
        !Fate ^ unsafe_elem(J) := uint16.cast_from_int(F),
        P1 = ( if F = 0 then P0 else P0 + 1 ),
        S1 = float.max(S0, Speed ^ unsafe_elem(I))
    ).

    % The largest speed of the NActive first active tiles.
    %
:- func fastest(int, array(int32), array(float)) = float.
:- mode fastest(in, array_ui, array_ui) = out is det.
fastest(NActive, Actives, Speed) = Fastest :-
    %$omp distribute parallel do reduction(max : Fastest)
    ( for(J, 0, NActive - 1), fromto(0.0, S0, S1, Fastest) do
        S1 = float.max(S0,
            Speed ^ unsafe_elem(int32.to_int(Actives ^ unsafe_elem(J))))
    ).

    % The directions in which the tile whose neighbours start at Base has
    % none.
    %
:- func missing(array(int32), int) = int.
:- mode missing(array_ui, in) = out is det.
missing(Nb, Base) =
    gap(Nb, Base, 0) \/ gap(Nb, Base, 1) \/ gap(Nb, Base, 2) \/
    gap(Nb, Base, 3) \/ gap(Nb, Base, 5) \/
    gap(Nb, Base, 6) \/ gap(Nb, Base, 7) \/ gap(Nb, Base, 8).

:- func gap(array(int32), int, int) = int.
:- mode gap(array_ui, in, in) = out is det.
gap(Nb, Base, D) = ( if Nb ^ unsafe_elem(Base + D) = 0i32 then 1 << D else 0 ).

    % The directions in which the tile whose neighbours start at Base has a
    % neighbour with a wet edge or corner facing it.
    %
:- func facing(array(int32), array(uint16), int) = int.
:- mode facing(array_ui, array_ui, in) = out is det.
facing(Nb, Edges, Base) =
    faces(Nb, Edges, Base, 0) \/ faces(Nb, Edges, Base, 1) \/
    faces(Nb, Edges, Base, 2) \/ faces(Nb, Edges, Base, 3) \/
    faces(Nb, Edges, Base, 5) \/ faces(Nb, Edges, Base, 6) \/
    faces(Nb, Edges, Base, 7) \/ faces(Nb, Edges, Base, 8).

:- func faces(array(int32), array(uint16), int, int) = int.
:- mode faces(array_ui, array_ui, in, in) = out is det.
faces(Nb, Edges, Base, D) = uint16.to_int(Edges ^ unsafe_elem(Neighbour)) /\
        (1 << (8 - D)) :-
    Neighbour = int32.to_int(Nb ^ unsafe_elem(Base + D)).

    % The directions from place P that are in the world: the rows' and the
    % columns' that are.
    %
:- func inside(geometry, int) = int.
inside(G, P) = Rows /\ Columns :-
    Side = G ^ side,
    Y = P / Side,
    X = P mod Side,
    Rows = ( if Y > 0 then 0x007 else 0 ) \/ 0x038 \/
        ( if Y < Side - 1 then 0x1c0 else 0 ),
    Columns = ( if X > 0 then 0x049 else 0 ) \/ 0x092 \/
        ( if X < Side - 1 then 0x124 else 0 ).

    % From the J'th active tile on, on the process's lane: each tile whose
    % fate ends it ends, its place in the actives, and its fate, taken by
    % the last one's.
    %
:- pred finish(int::in, array(uint16)::array_di, array(uint16)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, array(int32)::array_ui,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out) is det.
finish(J, !Fate, !Dir, Pos, !Nb, !Actives, !NActive, !Free, !NFree) :-
    ( if J >= !.NActive then
        true
    else if uint16.to_int(!.Fate ^ unsafe_elem(J)) /\ ends = 0 then
        finish(J + 1, !Fate, !Dir, Pos, !Nb, !Actives, !NActive, !Free,
            !NFree)
    else
        I = int32.to_int(!.Actives ^ unsafe_elem(J)),
        unlink(I, 0, !Nb),
        !Dir ^ unsafe_elem(int32.to_int(Pos ^ unsafe_elem(I))) := 0i32,
        !Free ^ unsafe_elem(!.NFree) := int32.cast_from_int(I),
        !:NFree = !.NFree + 1,
        !:NActive = !.NActive - 1,
        !Actives ^ unsafe_elem(J) := !.Actives ^ unsafe_elem(!.NActive),
        !Fate ^ unsafe_elem(J) := !.Fate ^ unsafe_elem(!.NActive),
        finish(J, !Fate, !Dir, Pos, !Nb, !Actives, !NActive, !Free, !NFree)
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

    % From the J'th active tile to the Kept'th, on the process's lane: each
    % direction of its fate starts a tile there, if none has started there
    % since the fate was decided.
    %
:- pred grow(geometry::in, int::in, int::in,
    array(uint16)::array_di, array(uint16)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out) is det.
grow(G, J, Kept, !Fate, !Dir, !Pos, !Nb, !Quiet, !Actives, !NActive, !Free,
        !NFree, !Overflow) :-
    ( if J >= Kept then
        true
    else
        Bits = uint16.to_int(!.Fate ^ unsafe_elem(J)) /\ 0x1ff,
        ( if Bits = 0 then
            true
        else
            I = int32.to_int(!.Actives ^ unsafe_elem(J)),
            grow_tile(G, I, 0, Bits, !Dir, !Pos, !Nb, !Quiet, !Actives,
                !NActive, !Free, !NFree, !Overflow)
        ),
        grow(G, J + 1, Kept, !Fate, !Dir, !Pos, !Nb, !Quiet, !Actives,
            !NActive, !Free, !NFree, !Overflow)
    ).

    % Tile I's directions from D on, of the bits Bits.
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

    % The tiles from tile row Y to row YLast, each from tile column XFirst
    % to XLast, by rows.
    %
:- pred first_tiles(geometry::in, int::in, int::in, int::in, int::in,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(int32)::array_di, array(int32)::array_uo,
    array(uint8)::array_di, array(uint8)::array_uo,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    array(int32)::array_di, array(int32)::array_uo, int::in, int::out,
    int::in, int::out) is det.
first_tiles(G, Y, YLast, XFirst, XLast, !Dir, !Pos, !Nb, !Quiet, !Actives,
        !NActive, !Free, !NFree, !Overflow) :-
    ( if Y > YLast then
        true
    else
        ( for(X, XFirst, XLast), fromto(!.Dir, D0, D1, Dir1),
            fromto(!.Pos, P0, P1, Pos1), fromto(!.Nb, N0, N1, Nb1),
            fromto(!.Quiet, Q0, Q1, Quiet1), fromto(!.Actives, A0, A1, Act1),
            fromto(!.NActive, C0, C1, NActive1),
            fromto(!.Free, F0, F1, Free1), fromto(!.NFree, R0, R1, NFree1),
            fromto(!.Overflow, O0, O1, Overflow1) do
            start(G, Y * G ^ side + X, D0, D1, P0, P1, N0, N1, Q0, Q1,
                A0, A1, C0, C1, F0, F1, R0, R1, O0, O1)
        ),
        first_tiles(G, Y + 1, YLast, XFirst, XLast, Dir1, !:Dir, Pos1, !:Pos,
            Nb1, !:Nb, Quiet1, !:Quiet, Act1, !:Actives, NActive1, !:NActive,
            Free1, !:Free, NFree1, !:NFree, Overflow1, !:Overflow)
    ).

%---------------------------------------------------------------------------%
% Loops across the teams: the first cells, new tiles' zeros, and the count.

    % The active tiles' cells at the start, and each one's largest speed.
    %
:- pred init(geometry::in, int::in, array(int32)::array_ui,
    array(int32)::array_ui, array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo) is det.
init(G, NActive, Actives, Pos, !Cur, !Speed) :-
    %$omp distribute
    ( for(J, 0, NActive - 1) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        init_tile(G, I, int32.to_int(Pos ^ unsafe_elem(I)), !Cur, Fastest),
        !Speed ^ unsafe_elem(I) := Fastest
    ).

:- pred init_tile(geometry::in, int::in, int::in,
    array(float)::array_di, array(float)::array_uo, float::out) is det.
init_tile(G, I, P, !Cur, Fastest) :-
    T = G ^ tile,
    Log = G ^ log,
    TT = T * T,
    Top = (P / G ^ side) << Log,
    Left = (P mod G ^ side) << Log,
    N = G ^ world,
    Case = G ^ case,
    %$omp parallel do reduction(max : Fastest)
    ( for(K, 0, TT - 1), fromto(0.0, S0, S1, Fastest) do
        H = start_depth(Top + (K >> Log), Left + (K /\ (T - 1)), N, Case),
        Cur1 = !.Cur ^ unsafe_elem(I * 3 * TT + K) := H,
        Cur2 = Cur1 ^ unsafe_elem(I * 3 * TT + TT + K) := 0.0,
        !:Cur = Cur2 ^ unsafe_elem(I * 3 * TT + 2 * TT + K) := 0.0,
        S1 = float.max(S0, speed(H, 0.0, 0.0))
    ).

    % The cells of the active tiles from the First'th on, which are new, all
    % dry, in the cells that the next step reads.
    %
:- pred zero(geometry::in, int::in, int::in, array(int32)::array_ui,
    array(float)::array_di, array(float)::array_uo) is det.
zero(G, First, NActive, Actives, !Cells) :-
    %$omp distribute
    ( for(J, First, NActive - 1) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        zero_tile(G, I, !Cells)
    ).

:- pred zero_tile(geometry::in, int::in,
    array(float)::array_di, array(float)::array_uo) is det.
zero_tile(G, I, !Cells) :-
    T = G ^ tile,
    TT = T * T,
    %$omp parallel do
    ( for(K, 0, 3 * TT - 1) do
        !Cells ^ unsafe_elem(I * 3 * TT + K) := 0.0
    ).

    % The wet cells, the checksum, the volume and, for Thacker's bowl, the
    % error's and the exact depths' sums at time Time, from each tile's
    % across its team.
    %
:- pred count(geometry::in, float::in, int::in, array(int32)::array_ui,
    array(int32)::array_ui, array(int32)::array_ui, array(float)::array_ui,
    int::out, int::out, float::out, float::out, float::out) is det.
count(G, Time, NActive, Actives, Pos, Wet, Cells, WetCells, Checksum, Volume,
        Error, Exact) :-
    array.init(int.max(NActive, 1), 0, Sums0),
    array.init(int.max(NActive, 1), 0.0, Volumes0),
    array.init(int.max(NActive, 1), 0.0, Errors0),
    array.init(int.max(NActive, 1), 0.0, Exacts0),
    %$omp distribute
    ( for(J, 0, NActive - 1), fromto(Sums0, S0, S1, Sums),
        fromto(Volumes0, V0, V1, Volumes), fromto(Errors0, E0, E1, Errors),
        fromto(Exacts0, X0, X1, Exacts) do
        I = int32.to_int(Actives ^ unsafe_elem(J)),
        tile_sums(G, Time, I, int32.to_int(Pos ^ unsafe_elem(I)), Cells, Sum,
            Vol, Err, Ex),
        S1 = S0 ^ unsafe_elem(J) := Sum,
        V1 = V0 ^ unsafe_elem(J) := Vol,
        E1 = E0 ^ unsafe_elem(J) := Err,
        X1 = X0 ^ unsafe_elem(J) := Ex
    ),
    ( for(J, 0, NActive - 1), fromto(0, W0, W1, WetCells),
        fromto(0, C0, C1, Checksum), fromto(0.0, V2, V3, Volume),
        fromto(0.0, E2, E3, Error), fromto(0.0, X2, X3, Exact) do
        W1 = W0 + int32.to_int(Wet ^ unsafe_elem(
            int32.to_int(Actives ^ unsafe_elem(J)))),
        C1 = C0 + Sums ^ unsafe_elem(J),
        V3 = V2 + Volumes ^ unsafe_elem(J),
        E3 = E2 + Errors ^ unsafe_elem(J),
        X3 = X2 + Exacts ^ unsafe_elem(J)
    ).

:- pred tile_sums(geometry::in, float::in, int::in, int::in,
    array(float)::array_ui, int::out, float::out, float::out, float::out)
    is det.
tile_sums(G, Time, I, P, Cells, Sum, Volume, Error, Exact) :-
    T = G ^ tile,
    Log = G ^ log,
    TT = T * T,
    Top = (P / G ^ side) << Log,
    Left = (P mod G ^ side) << Log,
    N = G ^ world,
    Case = G ^ case,
    %$omp parallel do reduction((+) : Sum) reduction((+) : Volume)
    %$omp& reduction((+) : Error) reduction((+) : Exact)
    ( for(K, 0, TT - 1), fromto(0, S0, S1, Sum), fromto(0.0, V0, V1, Volume),
        fromto(0.0, E0, E1, Error), fromto(0.0, X0, X1, Exact) do
        Y = Top + (K >> Log),
        X = Left + (K /\ (T - 1)),
        H = Cells ^ unsafe_elem(I * 3 * TT + K),
        HU = Cells ^ unsafe_elem(I * 3 * TT + TT + K),
        HV = Cells ^ unsafe_elem(I * 3 * TT + 2 * TT + K),
        S1 = S0 + (bits(H) + 3 * bits(HU) + 5 * bits(HV)) *
            (2 * (Y * N + X) + 1),
        V1 = V0 + H,
        ( if Case = 1 then
            Ex = thacker(Y, X, N, Time),
            E1 = E0 + float.abs(H - Ex),
            X1 = X0 + Ex
        else
            E1 = E0,
            X1 = X0
        )
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
    % bookkeeping's part of the average; the outcome; and the active tiles.
    %
:- pred results(int::in, array(float)::in, array(int)::in, int::in,
    float::in, int::in, float::in, int::in, float::in, float::in, int::in,
    io::di, io::uo) is det.
results(Steps, Times, Counts, Case, Time, Wet, Volume, Checksum, Error,
        Exact, Peak, !IO) :-
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
    write_results(Case, Time, Wet, Volume, Checksum,
        ( if Exact > 0.0 then Error / Exact else 0.0 ), Peak,
        Counts ^ elem(Steps - 1), Min, Max, Average,
        Keep / float(Steps - 1), !IO).

%---------------------------------------------------------------------------%
% What C does for the program: a float's bits, the clock, the environment
% and printf. The clock is libc's, on the device as on the host.

:- pragma foreign_decl("C", "#include <stdio.h>").
:- pragma foreign_decl("C", "#include <stdlib.h>").
:- pragma foreign_decl("C", "#include <string.h>").
:- pragma foreign_decl("C", "#include <time.h>").

    % The float's bits, -0.0 as 0.0.
    %
:- func bits(float) = int.
:- pragma foreign_proc("C",
    bits(F::in) = (B::out),
    [will_not_call_mercury, promise_pure],
"
    double y = F + 0.0;
    memcpy(&B, &y, sizeof y);
").

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

% Exits with status 1.
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
    header(World::in, Tile::in, Capacity::in, Case::in, Steps::in,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Shallow water, Mercury, GPU process, sparse tiles\\n\");
    printf(\"World: %ld x %ld cells, tiles of %ld x %ld, %ld in the pool\\n\",
        (long) World, (long) World, (long) Tile, (long) Tile,
        (long) Capacity - 1);
    printf(\"Case: %s\\n\", Case == 0 ? \"a column in a bowl\" :
        \"Thacker's bowl\");
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

:- pred write_results(int::in, float::in, int::in, float::in, int::in,
    float::in, int::in, int::in, float::in, float::in, float::in, float::in,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Case::in, Time::in, Wet::in, Volume::in, Checksum::in,
        Error::in, Peak::in, Final::in, Min::in, Max::in, Average::in,
        Keep::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Time: %.9f s\\n\", Time);
    printf(\"Wet cells: %ld\\n\", (long) Wet);
    printf(\"Volume: %.6e m^3\\n\", Volume);
    printf(\"Checksum: %016llx\\n\", (unsigned long long) Checksum);
    if (Case == 1)
        printf(\"Error: %.4e\\n\", Error);
    printf(\"Active tiles: %ld at most, %ld at the end\\n\", (long) Peak,
        (long) Final);
    printf(\"%-12s%-12s%-12s%-14s\\n\", \"Min (sec)\", \"Max\", \"Average\",
        \"Bookkeeping\");
    printf(\"%-12.6f%-12.6f%-12.6f%-14.6f\\n\", Min, Max, Average, Keep);
").
