% Processes and mailboxes, a Merkurius extension with no counterpart in the
% Mercury library (design document, Language extensions, "Processes and
% mailboxes"). So far only the mailboxes of one process: a mailbox is
% made, sent to and received from, and may be registered under a name, which
% a lookup finds, as in Erlang. A receive waits on one mailbox, or on several
% joined by a selector, as Gleam's Selector joins them.
%
% A handle names the node, by an id chosen at random when the program
% starts, and the place within it, 0 for the host, and carries a secret
% chosen at random when the mailbox is made, which a message from another
% node must give (design document, Handles, routing and places).
%
% A send copies the message, by its type_info, into a fragment of its own
% (runtime/mailbox.cpp), so the sender's cells are never shared with the
% receiver. A receive takes the oldest message its selector accepts, across
% all its mailboxes, by the order in which they arrived, as Erlang's receive
% takes the first message in the queue that any clause matches. It leaves
% the others queued in order, and waits until the timeout for one to arrive.
%
% Temporary (chunk 2, docs/INPROGRESS.md): spawn, close_mailbox's effect on
% other processes, monitors and request and reply come with processes; until then a program is one process, and
% a receive that waits with infinity for a message nobody can send waits
% forever.
%
% Nodes connect over TCP (runtime/node.cpp), after Erlang's distribution: a
% node listens on an address, another connects to it, and then each can
% look up the other's registered names and send to the other's mailboxes.
% A message to another node is encoded by its type and decoded by the
% mailbox's (design document, Wire format). A send to a node this one is not
% connected to, or whose connection is lost, is dropped, as one to a closed
% mailbox is. Two nodes connect only if both know the same cookie, which
% is never sent (runtime/node.cpp).

:- module process.
:- interface.
:- import_module io.
:- import_module maybe.

    % A mailbox that accepts messages of type M.
:- type mailbox(M).

    % A node, a program that mailboxes are on.
:- type node.

    % How long a receive waits: milliseconds, after(0) to poll.
:- type timeout
    --->    after(int)
    ;       infinity.

:- pred new_mailbox(mailbox(M)::out, io::di, io::uo) is det.

    % Drops the queued messages and the mailbox's name; later sends to the
    % mailbox are dropped, and a receive from it finds nothing.
:- pred close_mailbox(mailbox(M)::in, io::di, io::uo) is det.

:- pred send(mailbox(M)::in, M::in, io::di, io::uo) is det.

    % The oldest message for which the selector succeeds, mapped by it, or
    % no if none arrives before the timeout. A mailbox of another node
    % finds nothing.
:- pred receive(mailbox(M)::in, pred(M, R)::in(pred(in, out) is semidet),
    timeout::in, maybe(R)::out, io::di, io::uo) is det.

    % The selector that takes any message. It never fails, but is declared
    % semidet to fit receive's mode.
:- pred any(M::in, M::out) is semidet.

    % The cases of a receive over several mailboxes, each of which maps the
    % messages of its mailbox to R.
:- type selector(R).

    % The selector with no cases, whose receive finds nothing.
:- func new_selector = selector(R).

    % Adds a case that takes every message of the mailbox, mapped by the
    % function, so it never scans past its first.
:- func selecting(mailbox(M), func(M) = R, selector(R)) = selector(R).

    % Adds a case that takes the messages of the mailbox for which the
    % predicate succeeds, mapped by it.
:- func selecting_matching(mailbox(M)::in,
    pred(M, R)::in(pred(in, out) is semidet), selector(R)::in)
    = (selector(R)::out) is det.

    % The oldest message, across the selector's mailboxes, that one of its
    % cases accepts, mapped by that case, or no if none arrives before the
    % timeout. Of cases that accept the same message, the one added first
    % takes it. It finds nothing at once if every mailbox is closed or
    % another node's.
:- pred receive(selector(R)::in, timeout::in, maybe(R)::out,
    io::di, io::uo) is det.

    % The node this program is.
:- pred this_node(node::out, io::di, io::uo) is det.

    % The node a mailbox is on.
:- func node_of(mailbox(M)) = node.

    % Listens for other nodes on "host:port", where port 0 picks a free
    % port, which is returned. An empty host listens on every address.
:- pred listen(string::in, maybe_error(int)::out, io::di, io::uo) is det.

    % Connects to the node listening on "host:port".
:- pred connect(string::in, maybe_error(node)::out, io::di, io::uo) is det.

    % Registers a mailbox of this node under a name, which lookup finds, as
    % Erlang's register/2 does. It is an error if the name is taken, if the
    % mailbox has a name already, or if it is closed or another node's.
    % Closing the mailbox removes its name.
:- pred register(string::in, mailbox(M)::in, maybe_error::out,
    io::di, io::uo) is det.

    % Removes a name, if it is registered.
:- pred unregister(string::in, io::di, io::uo) is det.

    % The mailbox registered under a name on a node, if its messages are of
    % type M; otherwise an error saying that no mailbox has the name, or
    % that its messages are of another type. For another node, it is also
    % an error if this node is not connected to it, if the connection is
    % lost, or if the answer does not come within the timeout.
:- pred lookup(node::in, string::in, timeout::in,
    maybe_error(mailbox(M))::out, io::di, io::uo) is det.

    % For the expansion of the receive goal (design document, "The receive
    % goal"), not for programs. The selector of a receive goal of K arms
    % maps a message that the arm numbered I selects, from 0, to the arm's
    % variables, as receive_rest applied I times to receive_arm of them, or
    % for the last arm, K - 1 times to them; with no variables, they are
    % receive_none, and with several, a tuple.
:- type receive_arms(A, B)
    --->    receive_arm(A)
    ;       receive_rest(B).
:- type receive_none
    --->    receive_none.

    % receive/5, with a result whose type is process's, since the module
    % with the receive goal need not import maybe.
:- type receive_result(R)
    --->    receive_message(R)
    ;       receive_nothing.
:- pred receive_goal(selector(R)::in, timeout::in, receive_result(R)::out,
    io::di, io::uo) is det.

    % A receive goal without a timeout arm whose mailboxes are all closed.
:- pred receive_closed is erroneous.

:- implementation.
:- import_module bool.
:- import_module int.
:- import_module require.
:- import_module string.

    % The node's id, the place in the top 16 bits of a word whose other
    % 48 are the slot of the runtime's table, the generation that slot had
    % when the mailbox was made, which tells a closed mailbox's handle from
    % that of a later mailbox in the same slot, and the secret.
:- type mailbox(M)
    --->    mailbox(
                node_high   :: int,
                node_low    :: int,
                place_slot  :: int,
                generation  :: int,
                secret      :: int
            ).

    % The node's 128-bit id.
:- type node
    --->    node(int, int).

    % A selector's cases, the one added last first. The mailbox's type is
    % hidden, since each case has its own.
:- type selector(R)
    --->    no_cases
    ;       some [M] taking(mailbox(M), func(M) = R, selector(R))
    ;       some [M] matching(mailbox(M), func(M) = maybe(R), selector(R)).

    % Gives a foreign_proc the type_info of M when no argument of it has
    % type M.
:- type witness(M)
    --->    witness.

:- pragma foreign_decl("C", local, "
extern void merk_node(MR_Integer *high, MR_Integer *low);
extern void merk_mailbox_new(MR_Word type_info, MR_Integer *place_slot,
    MR_Integer *generation, MR_Integer *secret);
extern void merk_mailbox_close(MR_Integer node_high, MR_Integer node_low,
    MR_Integer place_slot, MR_Integer generation, MR_Integer secret);
extern void merk_mailbox_send(MR_Integer node_high, MR_Integer node_low,
    MR_Integer place_slot, MR_Integer generation, MR_Integer secret,
    MR_Word type_info, MR_Word message);
extern MR_Integer merk_mailbox_deadline(MR_Integer milliseconds);
extern MR_Integer merk_mailbox_wait(MR_Integer node_high,
    MR_Integer node_low, MR_Integer place_slot, MR_Integer generation,
    MR_Integer secret, MR_Integer position, MR_Integer deadline);
extern MR_Integer merk_mailbox_poll(MR_Integer node_high,
    MR_Integer node_low, MR_Integer place_slot, MR_Integer generation,
    MR_Integer secret, MR_Integer position);
extern MR_Integer merk_mailbox_arrivals(void);
extern MR_Integer merk_mailbox_last_stamp(void);
extern MR_Integer merk_mailbox_wait_arrival(MR_Integer seen,
    MR_Integer deadline);
extern MR_Word merk_mailbox_message(MR_Integer node_high,
    MR_Integer node_low, MR_Integer place_slot, MR_Integer generation,
    MR_Integer secret, MR_Integer position);
extern void merk_mailbox_remove(MR_Integer node_high, MR_Integer node_low,
    MR_Integer place_slot, MR_Integer generation, MR_Integer secret,
    MR_Integer position);
extern MR_Integer merk_mailbox_register(MR_Integer node_high,
    MR_Integer node_low, MR_Integer place_slot, MR_Integer generation,
    MR_Integer secret, MR_String name);
extern void merk_mailbox_unregister(MR_String name);
extern MR_Integer merk_mailbox_lookup(MR_String name, MR_Word type_info,
    MR_Integer *place_slot, MR_Integer *generation, MR_Integer *secret);
extern MR_Integer merk_node_listen(MR_String address, MR_Integer *port,
    const char **error);
extern MR_Integer merk_node_connect(MR_String address, MR_Integer *high,
    MR_Integer *low, const char **error);
extern MR_Integer merk_node_lookup(MR_Integer high, MR_Integer low,
    MR_String name, MR_Word type_info, MR_Integer milliseconds,
    MR_Integer *place_slot, MR_Integer *generation, MR_Integer *secret);
").

new_mailbox(Mailbox, !IO) :-
    this_node(node(High, Low), !IO),
    new_slot(witness : witness(M), PlaceSlot, Generation, Secret, !IO),
    Mailbox = mailbox(High, Low, PlaceSlot, Generation, Secret) : mailbox(M).

close_mailbox(mailbox(High, Low, PlaceSlot, Generation, Secret), !IO) :-
    close_slot(High, Low, PlaceSlot, Generation, Secret, !IO).

send(mailbox(High, Low, PlaceSlot, Generation, Secret), Message, !IO) :-
    send_message(High, Low, PlaceSlot, Generation, Secret, Message, !IO).

receive(Mailbox, Selector, Timeout, Result, !IO) :-
    deadline(milliseconds(Timeout), Deadline, !IO),
    scan(Mailbox, Selector, Deadline, 0, Result, !IO).

:- pragma foreign_proc("C",
    any(M::in, N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = M;
    SUCCESS_INDICATOR = MR_TRUE;
").

receive_closed :-
    error("receive: every mailbox is closed").

receive_goal(Selector, Timeout, Result, !IO) :-
    receive(Selector, Timeout, Received, !IO),
    (
        Received = yes(R),
        Result = receive_message(R)
    ;
        Received = no,
        Result = receive_nothing
    ).

new_selector = no_cases.

selecting(Mailbox, Map, Selector) = 'new taking'(Mailbox, Map, Selector).

selecting_matching(Mailbox, Selector, Rest) =
    'new matching'(Mailbox, try_selector(Selector), Rest).

receive(Selector, Timeout, Result, !IO) :-
    deadline(milliseconds(Timeout), Deadline, !IO),
    select(Selector, Deadline, starts(Selector), Result, !IO).

this_node(node(High, Low), !IO) :-
    node_id(High, Low, !IO).

node_of(Mailbox) = node(Mailbox ^ node_high, Mailbox ^ node_low).

register(Name, mailbox(High, Low, PlaceSlot, Generation, Secret), Result,
        !IO) :-
    register_slot(High, Low, PlaceSlot, Generation, Secret, Name, Status, !IO),
    ( if Status = 0 then
        Result = ok
    else if Status = 1 then
        Result = error("the name " ++ quoted(Name) ++ " is taken")
    else if Status = 2 then
        Result = error("the mailbox has a name already")
    else
        Result = error("the mailbox is closed or another node's")
    ).

unregister(Name, !IO) :-
    unregister_name(Name, !IO).

lookup(Node, Name, Timeout, Result, !IO) :-
    this_node(Self, !IO),
    Node = node(High, Low),
    ( if Node = Self then
        lookup_name(Name, witness : witness(M), Status, PlaceSlot, Generation,
            Secret, !IO)
    else
        lookup_remote(High, Low, Name, witness : witness(M),
            milliseconds(Timeout),
            Status, PlaceSlot, Generation, Secret, !IO)
    ),
    ( if Status = 0 then
        Result = ok(mailbox(High, Low, PlaceSlot, Generation, Secret)
            : mailbox(M))
    else if Status = 1 then
        Result = error("no mailbox is registered as " ++ quoted(Name))
    else if Status = 2 then
        Result = error("the mailbox registered as " ++ quoted(Name) ++
            " has messages of another type")
    else if Status = 3 then
        Result = error("not connected to the node")
    else if Status = 4 then
        Result = error("the connection to the node was lost")
    else
        Result = error("the node did not answer in time")
    ).

listen(Address, Result, !IO) :-
    listen_on(Address, Status, Port, Error, !IO),
    ( if Status = 0 then
        Result = ok(Port)
    else
        Result = error(Error)
    ).

connect(Address, Result, !IO) :-
    connect_to(Address, Status, High, Low, Error, !IO),
    ( if Status = 0 then
        Result = ok(node(High, Low))
    else
        Result = error(Error)
    ).

:- func quoted(string) = string.

quoted(Name) = "\"" ++ Name ++ "\"".

    % A timeout's milliseconds, -1 for infinity.
:- func milliseconds(timeout) = int.

milliseconds(infinity) = -1.
milliseconds(after(Milliseconds)) = max(Milliseconds, 0).

:- func try_selector(pred(M, R)::in(pred(in, out) is semidet), M::in)
    = (maybe(R)::out) is det.

try_selector(Selector, Message) = Result :-
    ( if Selector(Message, R) then
        Result = yes(R)
    else
        Result = no
    ).

    % For each case of a selector, the position in its mailbox from which
    % it has yet to look, the first case's first.
:- type starts
    --->    no_starts
    ;       start(int, starts).

:- func starts(selector(R)) = starts.

starts(no_cases) = no_starts.
starts(taking(_, _, Rest)) = start(0, starts(Rest)).
starts(matching(_, _, Rest)) = start(0, starts(Rest)).

    % What a case finds in its mailbox: the message it takes, mapped, with
    % the message's stamp, the mailbox and the message's position.
:- type found(R)
    --->    nothing
    ;       found(R, int, int, int, int, int, int, int).

    % The cases' oldest message, which it takes, waiting for the next
    % message to arrive in any mailbox until the deadline. The count of
    % arrivals is read before the cases look, so that one arriving while
    % they look is not missed. So is the last message's stamp: a message
    % found with a later stamp arrived while they looked, and an older one
    % may have arrived too, in a mailbox looked at before, so they look
    % again. The message found is then older than the mark, so the next
    % look takes it or an older one.
:- pred select(selector(R)::in, int::in, starts::in, maybe(R)::out,
    io::di, io::uo) is det.

select(Selector, Deadline, Starts0, Result, !IO) :-
    arrivals(Seen, !IO),
    last_stamp(Mark, !IO),
    look(Selector, Starts0, Starts, no, Open, nothing, Found, !IO),
    (
        Found = found(R, Stamp, High, Low, PlaceSlot, Generation, Secret,
            Position),
        ( if Stamp > Mark then
            select(Selector, Deadline, Starts, Result, !IO)
        else
            remove_message(High, Low, PlaceSlot, Generation, Secret,
                Position, !IO),
            Result = yes(R)
        )
    ;
        Found = nothing,
        (
            Open = no,
            Result = no
        ;
            Open = yes,
            wait_arrival(Seen, Deadline, Arrived, !IO),
            ( if Arrived = 0 then
                Result = no
            else
                select(Selector, Deadline, Starts, Result, !IO)
            )
        )
    ).

    % Each case looks in its mailbox from its start, which moves past the
    % messages it rejects; the oldest message found so far is kept. Open is
    % yes if some mailbox is open and this node's. The cases are visited the
    % last added first, so a later case that finds a message with the same
    % stamp, which is the same message, was added earlier and takes it.
:- pred look(selector(R)::in, starts::in, starts::out, bool::in, bool::out,
    found(R)::in, found(R)::out, io::di, io::uo) is det.

look(no_cases, Starts, Starts, !Open, !Found, !IO).
look(taking(Mailbox, Map, Rest), Starts0, Starts, !Open, !Found, !IO) :-
    Mailbox = mailbox(High, Low, PlaceSlot, Generation, Secret),
    split_starts(Starts0, Position, RestStarts0),
    poll(High, Low, PlaceSlot, Generation, Secret, Position, Stamp, !IO),
    ( if Stamp > 0 then
        !:Open = yes,
        message_at(High, Low, PlaceSlot, Generation, Secret, Position,
            Message, !IO),
        keep(found(Map(Message), Stamp, High, Low, PlaceSlot, Generation,
            Secret, Position), !Found)
    else if Stamp = 0 then
        !:Open = yes
    else
        true
    ),
    look(Rest, RestStarts0, RestStarts, !Open, !Found, !IO),
    Starts = start(Position, RestStarts).
look(matching(Mailbox, Try, Rest), Starts0, Starts, !Open, !Found, !IO) :-
    split_starts(Starts0, Position0, RestStarts0),
    scan_case(Mailbox, Try, Position0, Position, !Open, !Found, !IO),
    look(Rest, RestStarts0, RestStarts, !Open, !Found, !IO),
    Starts = start(Position, RestStarts).

    % From Position0 on, the first message that Try accepts, kept if it is
    % the oldest so far, with the position where a later look starts: that
    % message's, or the end of the queue.
:- pred scan_case(mailbox(M)::in, (func(M) = maybe(R))::in, int::in,
    int::out, bool::in, bool::out, found(R)::in, found(R)::out,
    io::di, io::uo) is det.

scan_case(Mailbox, Try, Position0, Position, !Open, !Found, !IO) :-
    Mailbox = mailbox(High, Low, PlaceSlot, Generation, Secret),
    poll(High, Low, PlaceSlot, Generation, Secret, Position0, Stamp, !IO),
    ( if Stamp > 0 then
        !:Open = yes,
        message_at(High, Low, PlaceSlot, Generation, Secret, Position0,
            Message, !IO),
        Selected = Try(Message),
        (
            Selected = yes(R),
            keep(found(R, Stamp, High, Low, PlaceSlot, Generation, Secret,
                Position0), !Found),
            Position = Position0
        ;
            Selected = no,
            scan_case(Mailbox, Try, Position0 + 1, Position, !Open, !Found,
                !IO)
        )
    else
        ( if Stamp = 0 then
            !:Open = yes
        else
            true
        ),
        Position = Position0
    ).

    % The first start and the rest; starts has one per case.
:- pred split_starts(starts::in, int::out, starts::out) is det.

split_starts(no_starts, 0, no_starts).
split_starts(start(Position, Rest), Position, Rest).

    % Keeps the older of two messages found, the new one if they have the
    % same stamp.
:- pred keep(found(R)::in, found(R)::in, found(R)::out) is det.

keep(New, Found0, Found) :-
    (
        Found0 = nothing,
        Found = New
    ;
        Found0 = found(_, Stamp0, _, _, _, _, _, _),
        ( if
            New = found(_, Stamp, _, _, _, _, _, _),
            Stamp =< Stamp0
        then
            Found = New
        else
            Found = Found0
        )
    ).

    % From the message at Position on, the first the selector accepts,
    % waiting for the next to arrive until the deadline. Only the
    % mailbox's owner removes messages, so the positions of those it has
    % passed stay as they are.
:- pred scan(mailbox(M)::in, pred(M, R)::in(pred(in, out) is semidet),
    int::in, int::in, maybe(R)::out, io::di, io::uo) is det.

scan(Mailbox, Selector, Deadline, Position, Result, !IO) :-
    Mailbox = mailbox(High, Low, PlaceSlot, Generation, Secret),
    wait_message(High, Low, PlaceSlot, Generation, Secret, Position, Deadline,
        Arrived, !IO),
    ( if Arrived = 0 then
        Result = no
    else
        message_at(High, Low, PlaceSlot, Generation, Secret, Position,
            Message, !IO),
        ( if Selector(Message, R) then
            remove_message(High, Low, PlaceSlot, Generation, Secret, Position,
                !IO),
            Result = yes(R)
        else
            scan(Mailbox, Selector, Deadline, Position + 1, Result, !IO)
        )
    ).

:- pred node_id(int::out, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    node_id(High::out, Low::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_node(&High, &Low);
").

    % A mailbox for messages of type M, which decodes them when they come
    % from another node.
:- pred new_slot(witness(M)::in, int::out, int::out, int::out,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    new_slot(_Witness::in, PlaceSlot::out, Generation::out, Secret::out,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_new(TypeInfo_for_M, &PlaceSlot, &Generation, &Secret);
").

:- pred close_slot(int::in, int::in, int::in, int::in, int::in,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    close_slot(High::in, Low::in, PlaceSlot::in, Generation::in, Secret::in,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_close(High, Low, PlaceSlot, Generation, Secret);
").

:- pred send_message(int::in, int::in, int::in, int::in, int::in, M::in,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    send_message(High::in, Low::in, PlaceSlot::in, Generation::in,
        Secret::in, Message::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_send(High, Low, PlaceSlot, Generation, Secret,
        TypeInfo_for_M, Message);
").

    % The time at which a receive stops waiting, -1 for never.
:- pred deadline(int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    deadline(Milliseconds::in, Deadline::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Deadline = merk_mailbox_deadline(Milliseconds);
").

    % 1 once the mailbox has a message at Position, 0 if the deadline
    % passes first, or the mailbox is closed or another node's.
:- pred wait_message(int::in, int::in, int::in, int::in, int::in, int::in,
    int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    wait_message(High::in, Low::in, PlaceSlot::in, Generation::in,
        Secret::in, Position::in, Deadline::in, Arrived::out,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Arrived = merk_mailbox_wait(High, Low, PlaceSlot, Generation, Secret,
        Position, Deadline);
").

    % The stamp of the message at Position, positive, or 0 if there is
    % none there yet, or -1 if the mailbox is closed or another node's.
:- pred poll(int::in, int::in, int::in, int::in, int::in, int::in,
    int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    poll(High::in, Low::in, PlaceSlot::in, Generation::in, Secret::in,
        Position::in, Stamp::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Stamp = merk_mailbox_poll(High, Low, PlaceSlot, Generation, Secret,
        Position);
").

    % The stamp of the last message queued.
:- pred last_stamp(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    last_stamp(Stamp::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Stamp = merk_mailbox_last_stamp();
").

    % The number of messages queued and mailboxes closed so far.
:- pred arrivals(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    arrivals(Count::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Count = merk_mailbox_arrivals();
").

    % 1 once a message is queued or a mailbox closed after Seen was read,
    % 0 if the deadline passes first.
:- pred wait_arrival(int::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    wait_arrival(Seen::in, Deadline::in, Arrived::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Arrived = merk_mailbox_wait_arrival(Seen, Deadline);
").

    % The message at Position, which wait_message or poll has found there.
:- pred message_at(int::in, int::in, int::in, int::in, int::in, int::in,
    M::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    message_at(High::in, Low::in, PlaceSlot::in, Generation::in, Secret::in,
        Position::in, Message::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    (void)TypeInfo_for_M;
    Message = merk_mailbox_message(High, Low, PlaceSlot, Generation, Secret,
        Position);
").

:- pred remove_message(int::in, int::in, int::in, int::in, int::in, int::in,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    remove_message(High::in, Low::in, PlaceSlot::in, Generation::in,
        Secret::in, Position::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_remove(High, Low, PlaceSlot, Generation, Secret, Position);
").

    % 0 if registered, 1 if the name is taken, 2 if the mailbox has a name,
    % 3 if it is closed or another node's.
:- pred register_slot(int::in, int::in, int::in, int::in, int::in,
    string::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    register_slot(High::in, Low::in, PlaceSlot::in, Generation::in,
        Secret::in, Name::in, Status::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Status = merk_mailbox_register(High, Low, PlaceSlot, Generation, Secret,
        Name);
").

:- pred unregister_name(string::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    unregister_name(Name::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_unregister(Name);
").

    % 0 and the mailbox's slot, generation and secret if its messages are
    % of type M, 1 if no mailbox has the name, 2 if they are of another.
:- pred lookup_name(string::in, witness(M)::in, int::out, int::out, int::out,
    int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    lookup_name(Name::in, _Witness::in, Status::out, PlaceSlot::out,
        Generation::out, Secret::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    PlaceSlot = 0;
    Generation = 0;
    Secret = 0;
    Status = merk_mailbox_lookup(Name, TypeInfo_for_M, &PlaceSlot,
        &Generation, &Secret);
").

    % As lookup_name, for another node, and 3 if this node is not connected
    % to it, 4 if the connection is lost, 5 if the answer does not come
    % within the milliseconds, if they are not negative.
:- pred lookup_remote(int::in, int::in, string::in, witness(M)::in, int::in,
    int::out, int::out, int::out, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    lookup_remote(High::in, Low::in, Name::in, _Witness::in, Milliseconds::in,
        Status::out, PlaceSlot::out, Generation::out, Secret::out,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    PlaceSlot = 0;
    Generation = 0;
    Secret = 0;
    Status = merk_node_lookup(High, Low, Name, TypeInfo_for_M, Milliseconds,
        &PlaceSlot, &Generation, &Secret);
").

:- pred listen_on(string::in, int::out, int::out, string::out,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    listen_on(Address::in, Status::out, Port::out, Error::out,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    const char *error = \"\";
    Port = 0;
    Status = merk_node_listen(Address, &Port, &error);
    Error = (MR_String) error;
").

:- pred connect_to(string::in, int::out, int::out, int::out, string::out,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    connect_to(Address::in, Status::out, High::out, Low::out, Error::out,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    const char *error = \"\";
    High = 0;
    Low = 0;
    Status = merk_node_connect(Address, &High, &Low, &error);
    Error = (MR_String) error;
").
