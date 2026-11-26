% mandel-recover.m's zoom, with its places on other nodes: OS processes on
% this machine, connected over TCP, which may be killed or cut off at any
% time (docs/INPROGRESS.md, Failure, step 6, D2). The frames, pixels and
% counts are mandel.m's, so the iterations and checksum are too, whichever
% nodes are lost and whenever.
%
% - The coordinator, `coordinator WORKERS`, listens on a free port of the
%   loopback, prints "listening PORT", and waits up to 30 seconds for
%   WORKERS workers to join. It then runs mandel-recover's coordinator,
%   with MANDEL_PLACES places (4), spawned round-robin through the
%   workers' agents.
% - A worker, `worker PORT`, connects to the coordinator, joins with its
%   agent's mailbox, and spawns places at its host when the coordinator
%   asks. It watches the coordinator: when the connection is lost, as the
%   coordinator quit, or it was cut off, it stops its places and ends, as
%   a node that was lost is fenced off and must start again to rejoin.
% - A place's down with noconnection means its node is lost: the
%   coordinator spawns no more places there, gives the place's chunk back
%   out, and spawns a new place on another live worker, or at its own node
%   when none is left, so the zoom always ends. A spawn whose request finds
%   its worker lost tries the next.
% - A chunk counts once, from the place it was given to, as in
%   mandel-recover.m, so a report from a place on a node that was cut off
%   is dropped.
%
% The output is the coordinator's: the iterations and checksum, the places
% and nodes lost, and the tiles given back. Tested with
% utils/faults/kill-nodes.py, which kills workers with SIGKILL, and with
% MERK_FAULTS partitions at the coordinator.
%
% Built with -fprocess-declarations and -fdo-loops. See README.md.

:- module mandel_nodes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list, maybe, process, string.

    % What a place is told.
:- type order
    --->    frame(int, float)
            % Frame F starts, of this width: ask for tiles.
    ;       tiles(int, int)
            % The first of the tiles to step and how many.
    ;       stop.

    % What a place tells the coordinator, as in mandel-recover.m.
:- type report
    --->    ask(int, int, int)
    ;       done(int, int, int, int, int, float).

    % What the coordinator reads: the places' reports, and their ends.
:- type happening
    --->    report(report)
    ;       lost(int, int, down_info).

    % What a worker's agent is told: spawn place P as incarnation Id, for
    % frames of Size pixels square in tiles of 2^Log, reporting to the
    % mailbox, and reply with its mailbox; or quit.
:- type agent_order
    --->    spawn_place(int, int, int, int, mailbox(report),
                mailbox(mailbox(order)))
    ;       quit.

    % What a worker joins with: its agent's mailbox.
:- type join
    --->    join(mailbox(agent_order)).

    % What a worker's agent reads: an order, or the coordinator's end.
:- type agent_input
    --->    agent_order(agent_order)
    ;       coordinator_lost(down_reason).

:- func as_report(report) = happening.
as_report(R) = report(R).

:- func same(happening) = happening.
same(H) = H.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if
        Args = ["coordinator", WorkersText],
        string.to_int(WorkersText, Workers),
        Workers >= 0
    then
        coordinator(Workers, !IO)
    else if Args = ["worker", Port] then
        worker(Port, !IO)
    else
        io.write_string("usage: coordinator WORKERS | worker PORT\n", !IO),
        exit_failure(!IO)
    ).

%---------------------------------------------------------------------------%
% A worker.

:- pred worker(string::in, io::di, io::uo) is det.
worker(Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(Coordinator),
        lookup(Coordinator, "joins", after(10000), Found, !IO),
        (
            Found = ok(Joins),
            new_mailbox(Orders, !IO),
            new_mailbox(Downs, !IO),
            monitor(Joins, Downs, reason, _, !IO),
            send(Joins, join(Orders), !IO),
            Selector = selecting(Orders, (func(O) = agent_order(O)),
                selecting(Downs, (func(D) = coordinator_lost(D)),
                    new_selector)),
            serve(Selector, [], !IO),
            close_mailbox(Downs, !IO),
            close_mailbox(Orders, !IO)
        ;
            Found = error(Error),
            io.write_string("worker: lookup: " ++ Error ++ "\n", !IO)
        )
    ;
        Connected = error(Error),
        io.write_string("worker: connect: " ++ Error ++ "\n", !IO)
    ).

    % Spawns places until told to quit or the coordinator is lost, and
    % then stops the places spawned.
:- pred serve(selector(agent_input)::in, list(mailbox(order))::in,
    io::di, io::uo) is det.
serve(Selector, Places, !IO) :-
    receive(Selector, infinity, Got, !IO),
    ( if Got = message(agent_order(spawn_place(P, Id, Size, Log, To,
            ReplyTo))) then
        spawn(host, place(To, P, Id, Size, Log), Box, !IO),
        send(ReplyTo, Box, !IO),
        serve(Selector, [Box | Places], !IO)
    else
        list.foldl(stop, Places, !IO)
    ).

%---------------------------------------------------------------------------%
% The coordinator.

:- type outcome
    --->    ok
    ;       failed(string).

    % Where a place is: through worker I's agent, or at the coordinator.
:- type location
    --->    at_worker(int)
    ;       at_coordinator.

    % What stays the same through the zoom: the coordinator's mailboxes,
    % the frame's size, its tiles' log, its tiles, and a chunk.
:- type setup
    --->    setup(mailbox(report), mailbox(happening), int, int, int, int).

:- type chunk
    --->    chunk(int, int).

    % A place: its mailbox, its incarnation, the chunk it was given and has
    % not reported, and where it is.
:- type slot
    --->    slot(mailbox(order), int, chunk, location).

    % Across the zoom: the places, by number; the workers' agents, with
    % whether each is live; the next incarnation; the restarts left; the
    % places lost and the tiles given back; and the iterations and
    % checksum.
:- type run
    --->    run(list(slot), list(agent), int, int, int, int, int, int).

:- type agent
    --->    agent(mailbox(agent_order), liveness).

    % Whether a worker's node is live, as far as the coordinator knows.
:- type liveness
    --->    live
    ;       lost.

    % Within a frame: the places waiting for tiles, the chunks given back,
    % the next tile not yet given, and the tiles stepped and their
    % iterations.
:- type frame
    --->    frame_state(list(int), list(chunk), int, int, int).

:- pred coordinator(int::in, io::di, io::uo) is det.
coordinator(Workers, !IO) :-
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
        new_mailbox(Joins, !IO),
        register("joins", Joins, _, !IO),
        listen("127.0.0.1:0", Listening, !IO),
        (
            Listening = ok(Port),
            io.write_string("listening " ++ int_to_string(Port) ++ "\n",
                !IO),
            io.flush_output(!IO),
            gather(Joins, Workers, [], Agents, !IO),
            PerRow = Size / Tile,
            new_mailbox(Reports, !IO),
            new_mailbox(Downs, !IO),
            Setup = setup(Reports, Downs, Size, Log, PerRow * PerRow, Chunk),
            Run0 = run([], Agents, 0, Restarts, 0, 0, 0, 0),
            list.foldl2(start_place(Setup), 0 .. K - 1, Run0, Run1, !IO),
            zoom(Setup, 0, Frames, 3.0, Run1, Run, ok, Outcome, !IO),
            Run = run(Slots, Final, _, _, Lost, Back, Total, Checksum),
            list.foldl(stop_place, Slots, !IO),
            list.foldl(quit_agent, Final, !IO),
            close_mailbox(Downs, !IO),
            close_mailbox(Reports, !IO),
            close_mailbox(Joins, !IO),
            (
                Outcome = ok,
                NodesLost = list.length(list.filter(is_lost, Final)),
                io.format("Iterations: %d\n", [i(Total)], !IO),
                io.format("Checksum: %d\n", [i(Checksum)], !IO),
                io.format("Places lost: %d\n", [i(Lost)], !IO),
                io.format("Nodes lost: %d\n", [i(NodesLost)], !IO),
                io.format("Tiles given back: %d\n", [i(Back)], !IO)
            ;
                Outcome = failed(Message),
                io.write_string(Message, !IO),
                exit_failure(!IO)
            )
        ;
            Listening = error(Error),
            io.write_string("listen: " ++ Error ++ "\n", !IO),
            exit_failure(!IO)
        )
    else
        io.write_string("MANDEL_TILE must be a power of two that divides " ++
            "MANDEL_SIZE, MANDEL_FRAMES, MANDEL_PLACES and MANDEL_CHUNK at " ++
            "least 1, and MANDEL_RESTARTS at least 0.\n", !IO),
        exit_failure(!IO)
    ).

    % The agents of up to N more workers that join within 30 seconds.
:- pred gather(mailbox(join)::in, int::in, list(agent)::in, list(agent)::out,
    io::di, io::uo) is det.
gather(Joins, N, !Agents, !IO) :-
    ( if N =< 0 then
        true
    else
        receive(Joins, any, after(30000), Got, !IO),
        (
            Got = message(join(Agent)),
            !:Agents = !.Agents ++ [agent(Agent, live)],
            gather(Joins, N - 1, !Agents, !IO)
        ;
            Got = timeout
        )
    ).

:- pred is_lost(agent::in) is semidet.
is_lost(agent(_, lost)).

:- pred quit_agent(agent::in, io::di, io::uo) is det.
quit_agent(agent(Box, Live), !IO) :-
    (
        Live = live,
        send(Box, quit, !IO)
    ;
        Live = lost
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

:- pred start_place(setup::in, int::in, run::in, run::out, io::di, io::uo)
    is det.
start_place(Setup, P, !Run, !IO) :-
    spawn_place(Setup, P, Slot, !Run, !IO),
    !.Run = run(Slots, Agents, Next, Restarts, Lost, Back, Total, Checksum),
    !:Run = run(Slots ++ [Slot], Agents, Next, Restarts, Lost, Back, Total,
        Checksum).

    % Place P as the next incarnation, monitored, on the live worker that
    % P's number picks, round-robin, or the next live one if that worker
    % is lost, or else at the coordinator.
:- pred spawn_place(setup::in, int::in, slot::out, run::in, run::out,
    io::di, io::uo) is det.
spawn_place(Setup, P, Slot, !Run, !IO) :-
    Setup = setup(Reports, Downs, Size, Log, _, _),
    !.Run = run(Slots, Agents0, Id, Restarts, Lost, Back, Total, Checksum),
    N = list.length(Agents0),
    ( if N > 0 then
        try_workers(Reports, P, Id, Size, Log, P mod N, N, Agents0, Agents,
            Spawned, !IO)
    else
        Agents = Agents0,
        Spawned = no
    ),
    (
        Spawned = yes({Box, I}),
        Where = at_worker(I)
    ;
        Spawned = no,
        spawn(host, place(Reports, P, Id, Size, Log), Box, !IO),
        Where = at_coordinator
    ),
    monitor(Box, Downs, (func(D) = lost(P, Id, D)), _, !IO),
    Slot = slot(Box, Id, chunk(0, 0), Where),
    !:Run = run(Slots, Agents, Id + 1, Restarts, Lost, Back, Total, Checksum).

    % Asks the live workers from worker I on, Left of them, to spawn the
    % place, marking each that is found lost.
:- pred try_workers(mailbox(report)::in, int::in, int::in, int::in, int::in,
    int::in, int::in, list(agent)::in, list(agent)::out,
    maybe({mailbox(order), int})::out, io::di, io::uo) is det.
try_workers(Reports, P, Id, Size, Log, I, Left, !Agents, Spawned, !IO) :-
    ( if Left =< 0 then
        Spawned = no
    else
        N = list.length(!.Agents),
        list.det_index0(!.Agents, I) = agent(Agent, Live),
        (
            Live = live,
            request(Agent, (func(R) = spawn_place(P, Id, Size, Log, Reports,
                R)), after(10000), Reply, !IO),
            ( if Reply = reply(Box) then
                Spawned = yes({Box, I})
            else
                mark_lost(I, !Agents),
                try_workers(Reports, P, Id, Size, Log, (I + 1) mod N,
                    Left - 1, !Agents, Spawned, !IO)
            )
        ;
            Live = lost,
            try_workers(Reports, P, Id, Size, Log, (I + 1) mod N, Left - 1,
                !Agents, Spawned, !IO)
        )
    ).

:- pred mark_lost(int::in, list(agent)::in, list(agent)::out) is det.
mark_lost(I, !Agents) :-
    list.det_index0(!.Agents, I) = agent(Agent, _),
    list.det_replace_nth(!.Agents, I + 1, agent(Agent, lost), !:Agents).

:- pred stop_place(slot::in, io::di, io::uo) is det.
stop_place(slot(Box, _, _, _), !IO) :-
    send(Box, stop, !IO).

    % Frames F to Frames of width Width on, each started and stepped.
:- pred zoom(setup::in, int::in, int::in, float::in, run::in, run::out,
    outcome::in, outcome::out, io::di, io::uo) is det.
zoom(Setup, F, Frames, Width, !Run, !Outcome, !IO) :-
    ( if F >= Frames ; !.Outcome \= ok then
        true
    else
        !.Run = run(Slots, _, _, _, _, _, _, _),
        list.foldl(start_frame(F, Width), Slots, !IO),
        Setup = setup(_, _, _, _, Tiles, _),
        step(Setup, F, Width, frame_state([], [], 0, 0, 0), Frame, !Run,
            !Outcome, !IO),
        Frame = frame_state(_, _, _, Stepped, Iterations),
        ( if !.Outcome = ok, Stepped \= Tiles then
            !:Outcome = failed(string.format("Frame %d: the places " ++
                "stepped %d tiles of %d.\n", [i(F), i(Stepped), i(Tiles)]))
        else
            true
        ),
        !.Run = run(S, A, N, R, L, B, Total, Checksum),
        !:Run = run(S, A, N, R, L, B, Total + Iterations,
            Checksum + (F + 1) * Iterations),
        zoom(Setup, F + 1, Frames, Width * 0.5, !Run, !Outcome, !IO)
    ).

:- pred start_frame(int::in, float::in, slot::in, io::di, io::uo) is det.
start_frame(F, Width, slot(Box, _, _, _), !IO) :-
    send(Box, frame(F, Width), !IO).

    % Answers the places' asks and reports, and their ends, until the
    % frame's tiles are stepped.
:- pred step(setup::in, int::in, float::in, frame::in, frame::out,
    run::in, run::out, outcome::in, outcome::out, io::di, io::uo) is det.
step(Setup, F, Width, !Frame, !Run, !Outcome, !IO) :-
    Setup = setup(Reports, Downs, _, _, Tiles, _),
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
    Setup = setup(_, _, _, _, Tiles, Chunk),
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
    !.Run = run(Slots, _, _, _, _, _, _, _),
    list.det_index0(Slots, P) = slot(Box, _, _, _),
    send(Box, tiles(First, Count), !IO),
    set_outstanding(P, Chunk, !Run).

    % Place P has ended: its chunk goes back, and to a waiting place if
    % there is one; if its node is lost, the node takes no more places; a
    % new place takes its number and starts the frame.
:- pred lose(setup::in, int::in, float::in, int::in, down_info::in,
    frame::in, frame::out, run::in, run::out, outcome::in, outcome::out,
    io::di, io::uo) is det.
lose(Setup, F, Width, P, Down, !Frame, !Run, !Outcome, !IO) :-
    chunk(First, Count) = outstanding(!.Run, P),
    !.Run = run(Slots0, Agents0, Next, Restarts, Lost, Back0, Total,
        Checksum),
    list.det_index0(Slots0, P) = slot(_, _, _, Where),
    ( if reason(Down) = noconnection, Where = at_worker(I) then
        mark_lost(I, Agents0, Agents)
    else
        Agents = Agents0
    ),
    !.Frame = frame_state(Waiting0, Back, Cursor, S, I0),
    list.delete_all(Waiting0, P, Waiting1),
    ( if Count > 0 then
        Back1 = [chunk(First, Count) | Back],
        Back2 = Back0 + Count
    else
        Back1 = Back,
        Back2 = Back0
    ),
    !:Frame = frame_state(Waiting1, Back1, Cursor, S, I0),
    ( if Restarts > 0 then
        !:Run = run(Slots0, Agents, Next, Restarts - 1, Lost + 1, Back2,
            Total, Checksum),
        spawn_place(Setup, P, Slot, !Run, !IO),
        Slot = slot(Box, _, _, _),
        !.Run = run(Slots1, Agents1, Next1, R1, L1, B1, T1, C1),
        list.det_replace_nth(Slots1, P + 1, Slot, Slots2),
        !:Run = run(Slots2, Agents1, Next1, R1, L1, B1, T1, C1),
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
current(run(Slots, _, _, _, _, _, _, _), P, Id) :-
    list.index0(Slots, P, slot(_, Id, _, _)).

:- func outstanding(run, int) = chunk.
outstanding(run(Slots, _, _, _, _, _, _, _), P) = Chunk :-
    list.det_index0(Slots, P) = slot(_, _, Chunk, _).

:- pred set_outstanding(int::in, chunk::in, run::in, run::out) is det.
set_outstanding(P, Chunk, !Run) :-
    !.Run = run(Slots, Agents, Next, Restarts, Lost, Back, Total, Checksum),
    list.det_index0(Slots, P) = slot(Box, Id, _, Where),
    list.det_replace_nth(Slots, P + 1, slot(Box, Id, Chunk, Where), Slots1),
    !:Run = run(Slots1, Agents, Next, Restarts, Lost, Back, Total, Checksum).

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

% Exits with status 1, since io.set_exit_status is not implemented yet.
:- pred exit_failure(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    exit_failure(_IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    fflush(stdout);
    exit(1);
").

