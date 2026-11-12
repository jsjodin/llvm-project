% Processes and mailboxes, a Merkurius extension with no counterpart in the
% Mercury library (design document, Language extensions, "Processes and
% mailboxes"). A process is spawned with a mailbox it owns; only a mailbox's
% owner receives from it, and it closes when its owner ends. The program
% ends when main does. A mailbox is
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
% Temporary (chunk 2, docs/INPROGRESS.md): monitors and request and reply are
% to come. A receive that waits with infinity for a message nobody can send
% waits forever.
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

    % A new process runs the predicate on a new mailbox, which it owns and
    % which is returned. It runs until the predicate returns.
:- pred spawn(pred(mailbox(M), io, io)::in(pred(in, di, uo) is det),
    mailbox(M)::out, io::di, io::uo) is det.

    % As spawn, at a place. A process kind written out with its arguments,
    % as in spawn(gpu(0), counter(0), Counter, !IO), spawns that kind there,
    % which on a GPU must be declared on gpu; the compiler makes such a call
    % one of the kind's own. Any other closure runs at the host, and at a
    % GPU is a software error, as a closure is the host's code.
:- pred spawn(place::in,
    pred(mailbox(M), io, io)::in(pred(in, di, uo) is det),
    mailbox(M)::out, io::di, io::uo) is det.

    % As spawn, with a stack of at least the given size in bytes, which
    % must be positive. It costs only address space until it is used.
:- pred spawn_with_stack(int::in,
    pred(mailbox(M), io, io)::in(pred(in, di, uo) is det),
    mailbox(M)::out, io::di, io::uo) is det.

    % Drops the queued messages and the mailbox's name; later sends to the
    % mailbox are dropped, and a receive from it is a run-time error.
    % Watches the process that owns the target mailbox: when it ends, or
    % at once if it has ended, the function makes a message from a
    % down_info, which goes to the watcher's mailbox. Across nodes it is
    % not supported yet.
:- type monitor_ref.
:- type down_reason.
:- type down_info
    --->    down_info(monitor_ref, down_reason).
:- pred monitor(mailbox(T)::in, mailbox(M)::in, (func(down_info) = M)::in,
    monitor_ref::out, io::di, io::uo) is det.

    % Stops watching. A message already queued stays.
:- pred demonitor(monitor_ref::in, io::di, io::uo) is det.

:- func reason(down_info) = down_reason.

    % "normal", "noproc" for a process that had ended already, or what
    % ended it, as the report of the failure says it. Temporary (chunk 2
    % step 6, docs/INPROGRESS.md): text, until exceptions can be caught
    % and a reason can carry the exception.
:- func reason_text(down_reason) = string.

    % Sends a request made by applying the function to a new mailbox for
    % the reply, which it returns, for the caller to add to its selector
    % and to close when it is done with it.
:- pred send_request(mailbox(Req)::in, (func(mailbox(Rep)) = Req)::in,
    mailbox(Rep)::out, io::di, io::uo) is det.

    % The result of a request.
:- type reply(Rep)
    --->    reply(Rep)
    ;       timeout
    ;       down(down_reason).

    % Sends a request and waits for its reply, for the timeout to pass, or
    % for the process that owns the server's mailbox to end, whichever
    % comes first. A reply that comes later goes nowhere. The server's
    % mailbox must be on this node, as a monitor's must, for now.
:- pred request(mailbox(Req)::in, (func(mailbox(Rep)) = Req)::in,
    timeout::in, reply(Rep)::out, io::di, io::uo) is det.

:- pred close_mailbox(mailbox(M)::in, io::di, io::uo) is det.

:- pred send(mailbox(M)::in, M::in, io::di, io::uo) is det.

    % What a receive gets: a message, or timeout if none arrives in time.
:- type received(R)
    --->    message(R)
    ;       timeout.

    % The oldest message for which the selector succeeds, mapped by it, or
    % timeout if none arrives in time. A mailbox that is closed or another
    % node's can never have a message for this receive, which is a mistake:
    % it is a run-time error, at once.
:- pred receive(mailbox(M)::in, pred(M, R)::in(pred(in, out) is semidet),
    timeout::in, received(R)::out, io::di, io::uo) is det.

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
    % cases accepts, mapped by that case, or timeout if none arrives in
    % time. Of cases that accept the same message, the one added first
    % takes it. Some mailboxes may be closed; it is a run-time error, at
    % once, if every one is closed or another node's, or there are none.
:- pred receive(selector(R)::in, timeout::in, received(R)::out,
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

    % The number of this process's team in a target teams region, from 0,
    % and how many teams the region has; 0 and 1 outside one. Without
    % OpenMP, the region runs once, as team 0 of 1 (docs/INPROGRESS.md,
    % Processes as teams).
:- pred team_number(int::out, io::di, io::uo) is det.
:- pred number_of_teams(int::out, io::di, io::uo) is det.

    % Waits until every team of the target teams region that this process is
    % a team of has called team_barrier as many times as this one has. What
    % a team did before its call, such as a send, has been done for every
    % team after its call returns. Outside a teams region, and without
    % OpenMP, there is one team, and it returns at once. The teams of a
    % region that calls it on a device must all run at once, as for team
    % mailboxes.
:- pred team_barrier(io::di, io::uo) is det.

    % The mailbox of team T for messages of type M, in the target teams
    % region that this process is a team of: one for each team and type,
    % made when first used, which lasts as long as the region. Any team of
    % the region sends to it, and only team T receives from it. Outside a
    % teams region, and without OpenMP, the only team is 0, whose mailboxes
    % last as long as the program. It is an error if T is not from 0 to the
    % number of teams less one. On a device, the messages must be of one
    % word so far.
:- pred team_mailbox(int::in, mailbox(M)::out, io::di, io::uo) is det.

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

    % The arm for a receive goal without a timeout arm, which has no time
    % limit: it is not reached, as receive/4 reports a goal with no open
    % mailbox.
:- pred receive_closed is erroneous.

    % Where a process runs: this node's host, or one of its GPUs, by its
    % number (design document, Handles, routing and places). spawn/5
    % spawns a process at a place; only a kind declared on gpu runs on a
    % GPU.
:- type place
    --->    host
    ;       gpu(int).

    % How many GPUs this node has for processes: those that LLVM's offload
    % library finds. With OpenMP's offloading they have OpenMP's numbers
    % for them, and otherwise they are numbered from 0.
:- pred gpus(int::out, io::di, io::uo) is det.

    % set_gpu_units(gpu(GPU), Percent, !IO): the GPU's processes run in one
    % kernel, a module's at a time, which holds every team slot of the
    % compute units it runs on until its processes have all ended. It takes
    % Percent of the units that the GPU's queues may use, from 1 to 100,
    % rounded up to a pair of units; the rest are for target regions and
    % other kernels, and a target region on a GPU whose processes' kernel
    % holds every unit is a software error. The default is 100, or
    % MERK_GPU_PROCESS_UNITS. It applies to the GPU's next kernel, so a
    % program sets it before its first spawn there, or between kernels, once
    % the processes of the last have all ended: while they run, as for the
    % host, which is not a GPU, it is a software error.
:- pred set_gpu_units(place::in, int::in, io::di, io::uo) is det.

    % For the expansion of process declarations (design document, "Processes
    % and mailboxes"), not for programs. A GPU process's kernel takes its
    % arguments as words: an int, a float or a character as one, and a
    % mailbox as its handle's five. gpu_words gives the words of the value
    % after those it is given.
:- type gpu_words.
:- func no_gpu_words = gpu_words.
:- typeclass gpu_argument(T) where [
    func gpu_words(T, gpu_words) = gpu_words
].
:- instance gpu_argument(int).
:- instance gpu_argument(float).
:- instance gpu_argument(character).
:- instance gpu_argument(mailbox(M)).

    % spawn_kernel(GPU, Spawn, Module, Kind, Words, Mailbox, !IO) spawns a
    % process of the module's kind, which runs on a GPU, on the GPU with the
    % number, with the words of its arguments and a new mailbox, which it
    % returns. A GPU that does not exist, or the module's image of its
    % kinds missing for the GPU, is a software error, which names the
    % predicate Spawn.
:- pred spawn_kernel(int::in, string::in, string::in, string::in,
    gpu_words::in, mailbox(M)::out, io::di, io::uo) is det.

    % The software error of spawning the kind, which runs on the host, at
    % gpu(GPU), which names the predicate Spawn.
:- pred spawn_host_kind_at_gpu(int::in, string::in, string::in,
    mailbox(M)::out, io::di, io::uo) is erroneous.

:- implementation.
:- import_module array.
:- import_module bool.
:- import_module char.
:- import_module int.
:- import_module list.
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

:- type monitor_ref
    --->    monitor_ref(int).

:- type down_reason
    --->    normal
    ;       no_process
    ;       failed(string).

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
#ifdef __AMDGPU__
extern void merk_device_team_barrier(void);
#else
extern void merk_team_barrier(void);
#endif
/* A team's number and the number of teams: on a device, its block's and
   the number of blocks, which are OpenMP's team number and number of teams
   in a region's kernel, and need no OpenMP in a GPU process's. On the host,
   OpenMP's when the program is linked with OpenMP's runtime; without it,
   the only team is 0. */
#ifdef __AMDGPU__
#include <gpuintrin.h>
#else
extern int omp_get_team_num(void) __attribute__((weak));
extern int omp_get_num_teams(void) __attribute__((weak));
#endif
extern MR_Integer merk_team_mailbox(MR_Integer team, MR_Word type_info,
    MR_Integer *place_slot, MR_Integer *generation, MR_Integer *secret);
extern void merk_mailbox_new(MR_Word type_info, MR_Integer *place_slot,
    MR_Integer *generation, MR_Integer *secret);
extern void merk_process_spawn(MR_Integer stack_bytes, MR_Word closure,
    MR_Word argument, MR_Integer node_high, MR_Integer node_low,
    MR_Integer place_slot, MR_Integer generation, MR_Integer secret);
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
extern MR_Integer merk_mailbox_read(MR_Integer node_high,
    MR_Integer node_low, MR_Integer place_slot, MR_Integer generation,
    MR_Integer secret, MR_Integer position, MR_Word *value, MR_Word *function,
    MR_Integer *monitor, const char **text);
extern void merk_mailbox_resolve(MR_Integer node_high, MR_Integer node_low,
    MR_Integer place_slot, MR_Integer generation, MR_Integer secret,
    MR_Integer position, MR_Word type_info, MR_Word value);
extern MR_Integer merk_monitor_new(MR_Integer target_high,
    MR_Integer target_low, MR_Integer target_place_slot,
    MR_Integer target_generation, MR_Integer target_secret,
    MR_Integer node_high, MR_Integer node_low, MR_Integer place_slot,
    MR_Integer generation, MR_Integer secret, MR_Word function);
extern void merk_monitor_remove(MR_Integer monitor);
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
extern MR_Integer merk_gpu_count(void);
extern const char *merk_gpu_set_units(MR_Integer gpu, MR_Integer percent);
extern const char *merk_gpu_process_spawn(MR_Word type_info, MR_Integer gpu,
    MR_String module, MR_String kind, MR_ArrayPtr words,
    MR_Integer *place_slot, MR_Integer *generation, MR_Integer *secret);
").

monitor(Target, Watcher, Function, monitor_ref(Id), !IO) :-
    Target = mailbox(TargetHigh, TargetLow, TargetPlaceSlot, TargetGeneration,
        TargetSecret),
    Watcher = mailbox(High, Low, PlaceSlot, Generation, Secret),
    new_monitor(TargetHigh, TargetLow, TargetPlaceSlot, TargetGeneration,
        TargetSecret, High, Low, PlaceSlot, Generation, Secret, Function, Id,
        !IO).

demonitor(monitor_ref(Id), !IO) :-
    remove_monitor(Id, !IO).

reason(down_info(_, Reason)) = Reason.

reason_text(normal) = "normal".
reason_text(no_process) = "noproc".
reason_text(failed(Text)) = Text.

send_request(Server, MakeRequest, ReplyTo, !IO) :-
    new_mailbox(ReplyTo, !IO),
    send(Server, MakeRequest(ReplyTo), !IO).

request(Server, MakeRequest, Timeout, Result, !IO) :-
    % The monitor comes first, so that a server that ends after the request
    % is sent cannot be missed.
    new_mailbox(Downs, !IO),
    monitor(Server, Downs, reason, Monitor, !IO),
    send_request(Server, MakeRequest, ReplyTo, !IO),
    Selector = selecting(ReplyTo, (func(R) = reply(R)),
        selecting(Downs, (func(D) = down(D)), new_selector)),
    receive(Selector, Timeout, Got, !IO),
    demonitor(Monitor, !IO),
    close_mailbox(ReplyTo, !IO),
    close_mailbox(Downs, !IO),
    (
        Got = message(Result)
    ;
        Got = timeout,
        Result = timeout
    ).

new_mailbox(Mailbox, !IO) :-
    this_node(node(High, Low), !IO),
    new_slot(witness : witness(M), PlaceSlot, Generation, Secret, !IO),
    Mailbox = mailbox(High, Low, PlaceSlot, Generation, Secret) : mailbox(M).

spawn(Body, Mailbox, !IO) :-
    start(0, Body, Mailbox, !IO).

spawn(Place, Body, Mailbox, !IO) :-
    (
        Place = host,
        spawn(Body, Mailbox, !IO)
    ;
        Place = gpu(GPU),
        error(string.format("process.spawn: a closure cannot run at " ++
            "gpu(%d): write the process kind with its arguments in the " ++
            "call, as in spawn(gpu(%d), kind(Arguments), Mailbox, !IO)",
            [i(GPU), i(GPU)]))
    ).

spawn_with_stack(Bytes, Body, Mailbox, !IO) :-
    ( if Bytes > 0 then
        start(Bytes, Body, Mailbox, !IO)
    else
        error("spawn_with_stack: the stack's size must be positive")
    ).

:- pred start(int::in, pred(mailbox(M), io, io)::in(pred(in, di, uo) is det),
    mailbox(M)::out, io::di, io::uo) is det.

start(Bytes, Body, Mailbox, !IO) :-
    new_mailbox(Mailbox, !IO),
    Mailbox = mailbox(High, Low, PlaceSlot, Generation, Secret),
    start_process(Bytes, Body, Mailbox, High, Low, PlaceSlot, Generation,
        Secret, !IO).

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

    % The words, the last first.
:- type gpu_words
    --->    gpu_words(list(int)).

no_gpu_words = gpu_words([]).

:- instance gpu_argument(int) where [
    gpu_words(N, gpu_words(Words)) = gpu_words([N | Words])
].
:- instance gpu_argument(float) where [
    gpu_words(X, gpu_words(Words)) = gpu_words([float_bits(X) | Words])
].
:- instance gpu_argument(character) where [
    gpu_words(C, gpu_words(Words)) = gpu_words([char.to_int(C) | Words])
].
:- instance gpu_argument(mailbox(M)) where [
    gpu_words(mailbox(High, Low, PlaceSlot, Generation, Secret),
        gpu_words(Words)) =
        gpu_words([Secret, Generation, PlaceSlot, Low, High | Words])
].

:- func float_bits(float) = int.
:- pragma foreign_proc("C",
    float_bits(X::in) = (Bits::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Bits = (MR_Integer) MR_float_to_word(X);
").

gpus(Count, !IO) :-
    gpu_count(Count, !IO).

spawn_kernel(GPU, Spawn, Module, Kind, gpu_words(Words), Mailbox, !IO) :-
    % The words are gathered the last first.
    Array = array.from_reverse_list(Words),
    launch_kernel(witness : witness(M), GPU, Module, Kind, Array, PlaceSlot,
        Generation, Secret, Error, !IO),
    ( if Error = "" then
        this_node(node(High, Low), !IO),
        Mailbox = mailbox(High, Low, PlaceSlot, Generation, Secret)
            : mailbox(M)
    else
        error(Spawn ++ ": " ++ Error)
    ).

spawn_host_kind_at_gpu(GPU, Spawn, Kind, _Mailbox, !IO) :-
    error(string.format("%s: the process kind %s runs on the host, not at " ++
        "gpu(%d): its declaration does not say on gpu", [s(Spawn), s(Kind),
        i(GPU)])).

set_gpu_units(Place, Percent, !IO) :-
    (
        Place = host,
        error("process.set_gpu_units: the host is not a GPU")
    ;
        Place = gpu(GPU),
        set_units(GPU, Percent, Error, !IO),
        ( if Error = "" then
            true
        else
            error("process.set_gpu_units: " ++ Error)
        )
    ).

:- pred set_units(int::in, int::in, string::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    set_units(GPU::in, Percent::in, Error::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    const char *error = merk_gpu_set_units(GPU, Percent);
    Error = (MR_String) (error ? error : \"\");
").

:- pred gpu_count(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    gpu_count(Count::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Count = merk_gpu_count();
").

:- pred launch_kernel(witness(M)::in, int::in, string::in, string::in,
    array(int)::in, int::out, int::out, int::out, string::out,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    launch_kernel(_Witness::in, GPU::in, Module::in, Kind::in, Words::in,
        PlaceSlot::out, Generation::out, Secret::out, Error::out,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    const char *error = merk_gpu_process_spawn(TypeInfo_for_M, GPU, Module,
        Kind, Words, &PlaceSlot, &Generation, &Secret);
    Error = (MR_String) (error ? error : \"\");
").

receive_goal(Selector, Timeout, Result, !IO) :-
    receive(Selector, Timeout, Received, !IO),
    (
        Received = message(R),
        Result = receive_message(R)
    ;
        Received = timeout,
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
:- pred select(selector(R)::in, int::in, starts::in, received(R)::out,
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
            Result = message(R)
        )
    ;
        Found = nothing,
        (
            Open = no,
            error("receive: every mailbox is closed or another node's")
        ;
            Open = yes,
            wait_arrival(Seen, Deadline, Arrived, !IO),
            ( if Arrived = 0 then
                Result = timeout
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
    int::in, int::in, received(R)::out, io::di, io::uo) is det.

scan(Mailbox, Selector, Deadline, Position, Result, !IO) :-
    Mailbox = mailbox(High, Low, PlaceSlot, Generation, Secret),
    wait_message(High, Low, PlaceSlot, Generation, Secret, Position, Deadline,
        Arrived, !IO),
    ( if Arrived < 0 then
        error("receive: the mailbox is closed or another node's")
    else if Arrived = 0 then
        Result = timeout
    else
        message_at(High, Low, PlaceSlot, Generation, Secret, Position,
            Message, !IO),
        ( if Selector(Message, R) then
            remove_message(High, Low, PlaceSlot, Generation, Secret, Position,
                !IO),
            Result = message(R)
        else
            scan(Mailbox, Selector, Deadline, Position + 1, Result, !IO)
        )
    ).

    % The block's on a device. On the host, OpenMP's when the program is
    % linked with its runtime, which a weak reference finds.
:- pragma foreign_proc("C",
    team_number(Number::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
#ifdef __AMDGPU__
    Number = __gpu_block_id_x();
#else
    Number = omp_get_team_num ? omp_get_team_num() : 0;
#endif
").

:- pragma foreign_proc("C",
    number_of_teams(Number::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
#ifdef __AMDGPU__
    Number = __gpu_num_blocks_x();
#else
    Number = omp_get_num_teams ? omp_get_num_teams() : 1;
#endif
").

:- pragma foreign_proc("C",
    team_barrier(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
#ifdef __AMDGPU__
    merk_device_team_barrier();
#else
    merk_team_barrier();
#endif
").

team_mailbox(Team, Mailbox, !IO) :-
    this_node(node(High, Low), !IO),
    team_slot(Team, witness : witness(M), Teams, PlaceSlot, Generation, Secret,
        !IO),
    ( if 0 =< Team, Team < Teams then
        Mailbox = mailbox(High, Low, PlaceSlot, Generation, Secret)
            : mailbox(M)
    else
        error("team_mailbox: the team number, " ++ int_to_string(Team) ++
            ", is not from 0 to " ++ int_to_string(Teams - 1))
    ).

    % The number of teams, and if the team is one of them, its mailbox for
    % messages of type M: its slot, generation and secret.
:- pred team_slot(int::in, witness(M)::in, int::out, int::out, int::out,
    int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    team_slot(Team::in, _Witness::in, Teams::out, PlaceSlot::out,
        Generation::out, Secret::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    PlaceSlot = 0;
    Generation = 0;
    Secret = 0;
    Teams = merk_team_mailbox(Team, TypeInfo_for_M, &PlaceSlot, &Generation,
        &Secret);
").

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

:- pred start_process(int::in,
    pred(mailbox(M), io, io)::in(pred(in, di, uo) is det), mailbox(M)::in,
    int::in, int::in, int::in, int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    start_process(Bytes::in, Body::in(pred(in, di, uo) is det), Mailbox::in,
        High::in, Low::in, PlaceSlot::in, Generation::in, Secret::in,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_process_spawn(Bytes, (MR_Word) Body, (MR_Word) Mailbox, High, Low,
        PlaceSlot, Generation, Secret);
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
    % passes first, or -1 if the mailbox is closed or another node's.
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
    % What a monitor left becomes the message here, by its function, so
    % that the function runs on the receiver's side, once.
:- pred message_at(int::in, int::in, int::in, int::in, int::in, int::in,
    M::out, io::di, io::uo) is det.

message_at(High, Low, PlaceSlot, Generation, Secret, Position, Message, !IO) :-
    read_message(High, Low, PlaceSlot, Generation, Secret, Position, Kind,
        Value, Function, Monitor, Text, !IO),
    ( if Kind = 0 then
        Message = Value
    else
        ( if Kind = 1 then
            Reason = normal
        else if Kind = 2 then
            Reason = no_process
        else
            Reason = failed(Text)
        ),
        Message = Function(down_info(monitor_ref(Monitor), Reason)),
        resolve_message(High, Low, PlaceSlot, Generation, Secret, Position,
            Message, !IO)
    ).

:- pred read_message(int::in, int::in, int::in, int::in, int::in, int::in,
    int::out, M::out, (func(down_info) = M)::out, int::out, string::out,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    read_message(High::in, Low::in, PlaceSlot::in, Generation::in, Secret::in,
        Position::in, Kind::out, Value::out, Function::out, Monitor::out,
        Text::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    const char *text;
    MR_Word function;
    (void)TypeInfo_for_M;
    Kind = merk_mailbox_read(High, Low, PlaceSlot, Generation, Secret,
        Position, (MR_Word *) &Value, &function, &Monitor, &text);
    Function = function;
    Text = (MR_String) text;
").

:- pred resolve_message(int::in, int::in, int::in, int::in, int::in, int::in,
    M::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    resolve_message(High::in, Low::in, PlaceSlot::in, Generation::in,
        Secret::in, Position::in, Message::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_resolve(High, Low, PlaceSlot, Generation, Secret, Position,
        TypeInfo_for_M, Message);
").

:- pred new_monitor(int::in, int::in, int::in, int::in, int::in, int::in,
    int::in, int::in, int::in, int::in, (func(down_info) = M)::in, int::out,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    new_monitor(TargetHigh::in, TargetLow::in, TargetPlaceSlot::in,
        TargetGeneration::in, TargetSecret::in, High::in, Low::in,
        PlaceSlot::in, Generation::in, Secret::in, Function::in, Id::out,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    (void)TypeInfo_for_M;
    Id = merk_monitor_new(TargetHigh, TargetLow, TargetPlaceSlot,
        TargetGeneration, TargetSecret, High, Low, PlaceSlot, Generation,
        Secret, (MR_Word) Function);
").

:- pred remove_monitor(int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    remove_monitor(Id::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_monitor_remove(Id);
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
