% Receives over several mailboxes joined by a selector (library/process.m):
%
% - The oldest message that some case accepts is taken, across mailboxes,
%   by the order of arrival, whatever the order of the cases.
% - Each case maps its mailbox's type into the selector's.
% - A selective case scans past what it rejects, which stays queued.
% - Of two cases that accept the same message, the one added first takes
%   it.
% - Erlang's two-step priority receive.
% - The empty selector, a selector whose mailboxes are all closed, and a
%   timeout find nothing.
%
% A model then checks a pseudo-random sequence of sends, closes, new
% mailboxes and receives over random selectors against a list of queues
% whose messages carry their order of arrival.
%
% mmc accepts the program with process.m, but has no process module to run
% it with, so the output is written by hand, and the model checks itself.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:oldest across mailboxes: [message(increment(1)), down("worker 1"), message(increment(2)), message(stop), down("worker 2"), none]
% OUTPUT-NEXT:cases in the other order: [message(increment(1)), down("worker 1"), message(increment(2)), message(stop), down("worker 2"), none]
% OUTPUT-NEXT:selective: [even(4), odd_word("three"), even(2), none]
% OUTPUT-NEXT:left queued: message(1)
% OUTPUT-NEXT:left queued: message("five")
% OUTPUT-NEXT:the case added first: message(first(6))
% OUTPUT-NEXT:the other case next: message(second(7))
% OUTPUT-NEXT:priority: [down("urgent"), message(increment(10)), message(stop)]
% OUTPUT-NEXT:timeout: timeout
% OUTPUT-NEXT:model: 30000 steps, 0 differ, 0 queues differ at the end
% OUTPUT-NEXT:stats(15096, 4070, 4853, 1502, 1498)
% OUTPUT-NOT:{{.}}

:- module process_selectors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module bool, int, list, maybe, process, string.

:- type counter_msg
    --->    increment(int)
    ;       stop.

:- type server_event
    --->    message(counter_msg)
    ;       down(string)
    ;       none.

:- func to_message(counter_msg) = server_event.
to_message(M) = message(M).

:- func to_down(string) = server_event.
to_down(D) = down(D).

    % Receives until nothing is left, polling.
:- pred receive_all(selector(R)::in, list(R)::out, io::di, io::uo) is det.
receive_all(Selector, Rs, !IO) :-
    receive(Selector, after(0), Got, !IO),
    (
        Got = message(R),
        receive_all(Selector, Rs0, !IO),
        Rs = [R | Rs0]
    ;
        Got = timeout,
        Rs = []
    ).

:- pred show(string::in, T::in, io::di, io::uo) is det.
show(What, X, !IO) :-
    io.write_string(What, !IO),
    io.write_string(": ", !IO),
    io.write(X, !IO),
    io.nl(!IO).

:- pred oldest(io::di, io::uo) is det.
oldest(!IO) :-
    new_mailbox(Self, !IO),
    new_mailbox(Downs, !IO),
    send(Self, increment(1), !IO),
    send(Downs, "worker 1", !IO),
    send(Self, increment(2), !IO),
    send(Self, stop, !IO),
    send(Downs, "worker 2", !IO),
    Sel = selecting(Self, to_message,
        selecting(Downs, to_down, new_selector)),
    receive_all(Sel, Events, !IO),
    show("oldest across mailboxes", Events ++ [none], !IO),
    send(Self, increment(1), !IO),
    send(Downs, "worker 1", !IO),
    send(Self, increment(2), !IO),
    send(Self, stop, !IO),
    send(Downs, "worker 2", !IO),
    Other = selecting(Downs, to_down,
        selecting(Self, to_message, new_selector)),
    receive_all(Other, OtherEvents, !IO),
    show("cases in the other order", OtherEvents ++ [none], !IO).

:- type parity
    --->    even(int)
    ;       odd_word(string)
    ;       none.

:- pred is_even(int::in, parity::out) is semidet.
is_even(N, even(N)) :-
    N mod 2 = 0.

:- pred is_three(string::in, parity::out) is semidet.
is_three(S, odd_word(S)) :-
    S = "three".

:- pred selective(io::di, io::uo) is det.
selective(!IO) :-
    new_mailbox(Numbers, !IO),
    new_mailbox(Words, !IO),
    send(Numbers, 1, !IO),
    send(Numbers, 4, !IO),
    send(Words, "five", !IO),
    send(Words, "three", !IO),
    send(Numbers, 2, !IO),
    Sel = selecting_matching(Numbers, is_even,
        selecting_matching(Words, is_three, new_selector)),
    receive_all(Sel, Got, !IO),
    show("selective", Got ++ [none], !IO),
    receive(Numbers, any, after(0), LeftNumber, !IO),
    show("left queued", LeftNumber, !IO),
    receive(Words, any, after(0), LeftWord, !IO),
    show("left queued", LeftWord, !IO).

:- type which
    --->    first(int)
    ;       second(int).

:- func take_first(int) = which.
take_first(N) = first(N).

:- pred match_second(int::in, which::out) is semidet.
match_second(N, second(N)) :-
    N > 0.

:- pred added_first(io::di, io::uo) is det.
added_first(!IO) :-
    new_mailbox(Box, !IO),
    send(Box, 6, !IO),
    send(Box, 7, !IO),
    % The case added first is the innermost.
    Sel = selecting_matching(Box, match_second,
        selecting(Box, take_first, new_selector)),
    receive(Sel, after(0), Got1, !IO),
    show("the case added first", Got1, !IO),
    Sel2 = selecting(Box, take_first,
        selecting_matching(Box, match_second, new_selector)),
    receive(Sel2, after(0), Got2, !IO),
    show("the other case next", Got2, !IO).

    % Erlang's two-step receive: an urgent message first, if there is one,
    % then the oldest.
:- pred next_event(selector(server_event)::in, mailbox(string)::in,
    server_event::out, io::di, io::uo) is det.
next_event(Sel, Downs, Event, !IO) :-
    receive(Downs, any, after(0), MaybeDown, !IO),
    (
        MaybeDown = message(D),
        Event = down(D)
    ;
        MaybeDown = timeout,
        receive(Sel, after(0), MaybeEvent, !IO),
        (
            MaybeEvent = message(Event)
        ;
            MaybeEvent = timeout,
            Event = none
        )
    ).

:- pred priority(io::di, io::uo) is det.
priority(!IO) :-
    new_mailbox(Self, !IO),
    new_mailbox(Downs, !IO),
    send(Self, increment(10), !IO),
    send(Self, stop, !IO),
    send(Downs, "urgent", !IO),
    Sel = selecting(Self, to_message, selecting(Downs, to_down, new_selector)),
    next_event(Sel, Downs, E1, !IO),
    next_event(Sel, Downs, E2, !IO),
    next_event(Sel, Downs, E3, !IO),
    show("priority", [E1, E2, E3], !IO).

:- pred nothing(io::di, io::uo) is det.
nothing(!IO) :-
    new_mailbox(C, !IO),
    receive(selecting(C, (func(N) = N + 1), new_selector), after(50),
        TimedOut, !IO),
    show("timeout", TimedOut, !IO).

%---------------------------------------------------------------------------%
% The model: each mailbox, its handle, whether it is open, and its queue,
% oldest first, of each message with its stamp, the number of messages
% sent before it.

:- type seed == int.

:- pred next(int::in, int::out, seed::in, seed::out) is det.
next(N, X, S0, S) :-
    S = (S0 * 1103515245 + 12345) mod 2147483648,
    X = (S // 65536) mod N.

:- type queued ---> queued(int, int).
:- type entry ---> entry(mailbox(int), bool, list(queued)).

    % A case of a random selector: which mailbox, and whether it takes every
    % message or those divisible by the divisor.
:- type case ---> case(int, maybe(int)).

:- type stats
    --->    stats(
                sends :: int,
                got :: int,
                none :: int,
                closes :: int,
                opens :: int
            ).

    % A case's result: the message, times 10, plus the number of the case,
    % counted from the outermost, the last added, so that the case that took
    % it shows.
:- func tag(int, int) = int.
tag(Case, V) = V * 10 + Case.

:- func taking_case(int, int) = int.
taking_case(Case, V) = tag(Case, V).

:- pred matching_case(int::in, int::in, int::in, int::out) is semidet.
matching_case(D, Case, V, R) :-
    V mod D = 0,
    R = tag(Case, V).

    % The selector of the cases, the last added first.
:- func selector(list(entry), int, list(case)) = selector(int).
selector(_, _, []) = new_selector.
selector(Model, Number, [case(I, Divisor) | Cases]) = Sel :-
    entry(Box, _, _) = det_index0(Model, I),
    Rest = selector(Model, Number + 1, Cases),
    (
        Divisor = no,
        Sel = selecting(Box, taking_case(Number), Rest)
    ;
        Divisor = yes(D),
        Sel = selecting_matching(Box, matching_case(D, Number), Rest)
    ).

    % The message the model expects, as the case found it, its stamp, the
    % mailbox and the rest of its queue.
:- type expected ---> expected(int, int, int, list(queued)).

:- pred first_divisible(int::in, list(queued)::in, queued::out,
    list(queued)::out) is semidet.
first_divisible(D, [Q | Qs], Found, Rest) :-
    Q = queued(_, V),
    ( if V mod D = 0 then
        Found = Q,
        Rest = Qs
    else
        first_divisible(D, Qs, Found, Rest0),
        Rest = [Q | Rest0]
    ).

    % The cases are given the last added first. The earlier added are
    % looked at first, and a later added case replaces the message found so
    % far only with an older one: the case added first takes a message that
    % two accept.
:- pred expect(list(entry)::in, int::in, list(case)::in,
    maybe(expected)::in, maybe(expected)::out) is det.
expect(_, _, [], !Best).
expect(Model, Number, [case(I, Divisor) | Cases], !Best) :-
    expect(Model, Number + 1, Cases, !Best),
    entry(_, Open, Queue) = det_index0(Model, I),
    ( if
        Open = yes,
        (
            Divisor = no,
            Queue = [Found | Rest]
        ;
            Divisor = yes(D),
            first_divisible(D, Queue, Found, Rest)
        ),
        Found = queued(Stamp, V),
        (
            !.Best = no
        ;
            !.Best = yes(expected(_, BestStamp, _, _)),
            Stamp < BestStamp
        )
    then
        !:Best = yes(expected(tag(Number, V), Stamp, I, Rest))
    else
        true
    ).

:- pred random_cases(int::in, int::in, list(case)::out,
    seed::in, seed::out) is det.
random_cases(Boxes, N, Cases, !S) :-
    ( if N =< 0 then
        Cases = []
    else
        next(Boxes, I, !S),
        next(3, Kind, !S),
        ( if Kind = 0 then
            Divisor = no
        else
            next(4, D0, !S),
            Divisor = yes(D0 + 2)
        ),
        random_cases(Boxes, N - 1, Cases0, !S),
        Cases = [case(I, Divisor) | Cases0]
    ).

:- pred replace(int::in, T::in, list(T)::in, list(T)::out) is det.
replace(_, _, [], []).
replace(I, X, [Y | Ys], [Z | Zs]) :-
    ( if I = 0 then
        Z = X,
        Zs = Ys
    else
        Z = Y,
        replace(I - 1, X, Ys, Zs)
    ).

:- pred step(int::in, list(entry)::in, list(entry)::out, int::in, int::out,
    int::in, int::out, stats::in, stats::out, seed::in, seed::out,
    io::di, io::uo) is det.
step(Step, !Model, !Stamp, !Bad, !Stats, !S, !IO) :-
    next(10, Op, !S),
    next(length(!.Model), I, !S),
    entry(Box, Open, Queue) = det_index0(!.Model, I),
    ( if Op =< 4 then
        next(100, V, !S),
        send(Box, V, !IO),
        ( if Open = yes then
            replace(I, entry(Box, Open, Queue ++ [queued(!.Stamp, V)]),
                !Model),
            !:Stamp = !.Stamp + 1
        else
            true
        ),
        !Stats ^ sends := !.Stats ^ sends + 1
    else if Op =< 7 then
        next(4, N, !S),
        random_cases(length(!.Model), N + 1, Cases, !S),
        % With no open mailbox, a receive is an error.
        ( if
            list.member(case(K, _), Cases),
            entry(_, OpenK, _) = det_index0(!.Model, K),
            OpenK = yes
        then
            receive(selector(!.Model, 0, Cases), after(0), Got, !IO)
        else
            Got = timeout
        ),
        expect(!.Model, 0, Cases, no, Best),
        (
            Best = yes(expected(R, _, J, Rest)),
            Expected = message(R),
            entry(BoxJ, OpenJ, _) = det_index0(!.Model, J),
            replace(J, entry(BoxJ, OpenJ, Rest), !Model),
            !Stats ^ got := !.Stats ^ got + 1
        ;
            Best = no,
            Expected = timeout,
            !Stats ^ none := !.Stats ^ none + 1
        ),
        ( if Expected = Got then
            true
        else
            !:Bad = !.Bad + 1,
            ( if !.Bad =< 5 then
                io.write({Step, Cases, Expected, Got}, !IO),
                io.nl(!IO)
            else
                true
            )
        )
    else if Op = 8 then
        ( if Open = yes then
            close_mailbox(Box, !IO),
            replace(I, entry(Box, no, []), !Model),
            !Stats ^ closes := !.Stats ^ closes + 1
        else
            true
        )
    else
        ( if Open = no then
            new_mailbox(New, !IO),
            replace(I, entry(New, yes, []), !Model),
            !Stats ^ opens := !.Stats ^ opens + 1
        else
            true
        )
    ).

:- pred steps(int::in, int::in, list(entry)::in, list(entry)::out,
    int::in, int::out, int::in, int::out, stats::in, stats::out,
    seed::in, seed::out, io::di, io::uo) is det.
steps(Step, N, !Model, !Stamp, !Bad, !Stats, !S, !IO) :-
    ( if Step >= N then
        true
    else
        step(Step, !Model, !Stamp, !Bad, !Stats, !S, !IO),
        steps(Step + 1, N, !Model, !Stamp, !Bad, !Stats, !S, !IO)
    ).

    % At the end, each open mailbox must hold exactly its model queue.
:- pred drain(list(entry)::in, int::in, int::out, io::di, io::uo) is det.
drain([], !Left, !IO).
drain([entry(Box, Open, Queue) | Es], !Left, !IO) :-
    (
        Open = yes,
        receive_all(selecting(Box, (func(V) = V), new_selector), Got, !IO),
        ( if Got = map(queued_value, Queue) then
            true
        else
            !:Left = !.Left + 1
        )
    ;
        Open = no
    ),
    drain(Es, !Left, !IO).

:- func queued_value(queued) = int.
queued_value(queued(_, V)) = V.

:- pred new_entries(int::in, list(entry)::out, io::di, io::uo) is det.
new_entries(N, Es, !IO) :-
    ( if N =< 0 then
        Es = []
    else
        new_mailbox(Box, !IO),
        new_entries(N - 1, Es0, !IO),
        Es = [entry(Box, yes, []) | Es0]
    ).

:- pred check_model(io::di, io::uo) is det.
check_model(!IO) :-
    new_entries(5, Model0, !IO),
    Steps = 30000,
    steps(0, Steps, Model0, Model, 0, _, 0, Bad, stats(0, 0, 0, 0, 0), Stats,
        2026, _, !IO),
    drain(Model, 0, Left, !IO),
    io.write_string("model: ", !IO),
    io.write_int(Steps, !IO),
    io.write_string(" steps, ", !IO),
    io.write_int(Bad, !IO),
    io.write_string(" differ, ", !IO),
    io.write_int(Left, !IO),
    io.write_string(" queues differ at the end\n", !IO),
    % How much of each kind of step it took, so that a change to the
    % generator that stops covering one shows.
    io.write(Stats, !IO),
    io.nl(!IO).

main(!IO) :-
    oldest(!IO),
    selective(!IO),
    added_first(!IO),
    priority(!IO),
    nothing(!IO),
    check_model(!IO).
