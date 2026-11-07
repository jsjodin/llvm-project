% Self-checking tests of mailboxes, for which mmc cannot be the oracle,
% as it has no process module: the program checks itself, and prints what
% it checked and how many checks failed.
%
% - Round trips: pseudo-random terms of a type with every kind of value a
%   message can hold are sent, queued together, received and compared
%   with the originals, by unification and by compare; in monomorphic
%   code and through type_infos in polymorphic code.
% - No sharing: after a send, the sender's array, string and cells are
%   overwritten in place by foreign_procs, which no Mercury program can
%   do; the received message must still have the old values.
% - A model: a pseudo-random sequence of sends, receives with and without
%   selectors, closes, new mailboxes and sends to stale handles, each
%   checked against a pure model of the mailboxes, a list of queues.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:terms: 3000 checked, 0 differ
% OUTPUT-NEXT:terms through a type_info: 3000 checked, 0 differ
% OUTPUT-NEXT:pairs of maybe lists and tuples: 3000 checked, 0 differ
% OUTPUT-NEXT:a list of 300000 terms: 1 checked, 0 differ
% OUTPUT-NEXT:string: copied
% OUTPUT-NEXT:array: copied
% OUTPUT-NEXT:cells in a list, an array and a tuple: copied
% OUTPUT-NEXT:a received array sent again: copied
% OUTPUT-NEXT:model: 40000 steps, 0 differ, 0 queues differ at the end
% OUTPUT-NEXT:stats(13330, 3357, 2107, 1570, 12986, 1694, 1691, 0)

:- module process_checks.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, bool, char, float, int, int8, list, maybe, pair,
    process.

%---------------------------------------------------------------------------%
% Pseudo-random numbers.

:- type seed == int.

:- pred next(int::in, int::out, seed::in, seed::out) is det.
next(Bound, N, S0, S) :-
    S = (S0 * 1103515245 + 12345) /\ 2147483647,
    N = (S >> 8) mod Bound.

%---------------------------------------------------------------------------%
% Terms with every kind of value a message can hold: integers, floats,
% strings, characters, narrow integers, an enum, a notag of a float and
% one of a term, tuples, lists, arrays, polymorphic types, and more
% constructors with arguments than there are tags, so that the last tag
% is shared and those cells have secondary tags.

:- type t
    --->    leaf
    ;       num(int)
    ;       real(float)
    ;       text(string)
    ;       ch(char)
    ;       pr(t, t)
    ;       many(list(t))
    ;       box(wrapped)
    ;       tup({int, t})
    ;       arr(array(t))
    ;       opt(maybe(t))
    ;       hue(colour)
    ;       small(int8)
    ;       temp(celsius).

:- type wrapped ---> wrapped(t).
:- type celsius ---> celsius(float).
:- type colour ---> red ; green ; blue.

:- pred gen(int::in, t::out, seed::in, seed::out) is det.
gen(Depth, T, !S) :-
    next(( if Depth =< 0 then 7 else 14 ), K, !S),
    ( if K = 0 then
        T = leaf
    else if K = 1 then
        next(2000001, N, !S),
        T = num(N - 1000000)
    else if K = 2 then
        next(1000, N, !S),
        T = real(float(N - 500) / 8.0)
    else if K = 3 then
        next(5, N, !S),
        T = text(det_index0(["", "a", "λx", "two words", "\"quoted\""], N))
    else if K = 4 then
        next(4, N, !S),
        T = ch(det_index0(['a', 'Z', 'λ', '\n'], N))
    else if K = 5 then
        next(3, N, !S),
        T = hue(det_index0([red, green, blue], N))
    else if K = 6 then
        next(3, N, !S),
        T = small(det_index0([-128i8, 0i8, 127i8], N))
    else if K = 7 then
        gen(Depth - 1, A, !S),
        gen(Depth - 1, B, !S),
        T = pr(A, B)
    else if K = 8 then
        next(5, N, !S),
        gen_list(N, Depth - 1, L, !S),
        T = many(L)
    else if K = 9 then
        gen(Depth - 1, A, !S),
        T = box(wrapped(A))
    else if K = 10 then
        next(100, N, !S),
        gen(Depth - 1, A, !S),
        T = tup({N, A})
    else if K = 11 then
        next(4, N, !S),
        gen_list(N, Depth - 1, L, !S),
        T = arr(array(L))
    else if K = 12 then
        next(2, N, !S),
        gen(Depth - 1, A, !S),
        T = opt(( if N = 0 then no else yes(A) ))
    else
        next(400, N, !S),
        T = temp(celsius(float(N) / 4.0))
    ).

:- pred gen_list(int::in, int::in, list(t)::out, seed::in, seed::out) is det.
gen_list(N, Depth, L, !S) :-
    ( if N =< 0 then
        L = []
    else
        gen(Depth, H, !S),
        gen_list(N - 1, Depth, T, !S),
        L = [H | T]
    ).

:- pred gen_terms(int::in, list(t)::out, seed::in, seed::out) is det.
gen_terms(N, L, !S) :-
    gen_list(N, 5, L, !S).

%---------------------------------------------------------------------------%
% Round trips.

:- pred send_each(mailbox(T)::in, list(T)::in, io::di, io::uo) is det.
send_each(_, [], !IO).
send_each(Box, [X | Xs], !IO) :-
    send(Box, X, !IO),
    send_each(Box, Xs, !IO).

    % Receives as many messages as there are originals and counts those
    % that differ from theirs, by unification or by compare, and those
    % missing; then checks that nothing is left.
:- pred check_each(mailbox(T)::in, list(T)::in, int::in, int::out,
    io::di, io::uo) is det.
check_each(Box, [], !Bad, !IO) :-
    receive(Box, any, after(0), Left, !IO),
    ( if Left = yes(_) then !:Bad = !.Bad + 1 else true ).
check_each(Box, [X | Xs], !Bad, !IO) :-
    receive(Box, any, after(0), M, !IO),
    ( if
        M = yes(Y),
        X = Y,
        compare((=), X, Y)
    then
        true
    else
        !:Bad = !.Bad + 1
    ),
    check_each(Box, Xs, !Bad, !IO).

    % The same in code polymorphic in the message type, whose sends copy
    % through the type_info that it is passed.
:- pred round_trips(list(T)::in, int::out, io::di, io::uo) is det.
round_trips(Xs, Bad, !IO) :-
    new_mailbox(Box, !IO),
    send_each(Box, Xs, !IO),
    check_each(Box, Xs, 0, Bad, !IO),
    close_mailbox(Box, !IO).

:- pred report(string::in, int::in, int::in, io::di, io::uo) is det.
report(What, Count, Bad, !IO) :-
    io.write_string(What, !IO),
    io.write_string(": ", !IO),
    io.write_int(Count, !IO),
    io.write_string(" checked, ", !IO),
    io.write_int(Bad, !IO),
    io.write_string(" differ\n", !IO).

:- pred check_round_trips(io::di, io::uo) is det.
check_round_trips(!IO) :-
    gen_terms(3000, Terms, 12345, _),
    % Monomorphic: a mailbox of t.
    new_mailbox(Box, !IO),
    send_each(Box, Terms, !IO),
    check_each(Box, Terms, 0, Bad1, !IO),
    report("terms", length(Terms), Bad1, !IO),
    % Polymorphic, at t and at types made from it.
    round_trips(Terms, Bad2, !IO),
    report("terms through a type_info", length(Terms), Bad2, !IO),
    Wrapped = list.map((func(T) = yes([T, leaf]) - {T, "x"}), Terms),
    round_trips(Wrapped, Bad3, !IO),
    report("pairs of maybe lists and tuples", length(Wrapped), Bad3, !IO),
    % One long message, a list of all the terms, many times over.
    Long = repeat(100, Terms, []),
    round_trips([Long], Bad4, !IO),
    report("a list of 300000 terms", 1, Bad4, !IO).

:- func repeat(int, list(T), list(T)) = list(T).
repeat(N, Xs, Acc) =
    ( if N =< 0 then Acc else repeat(N - 1, Xs, Xs ++ Acc) ).

%---------------------------------------------------------------------------%
% No sharing: the message is a copy. Each foreign_proc overwrites a value
% the sender has already sent, in place, as the representation has it
% (design document, "Type representation").

:- type cell ---> cell(int, int).

:- pragma foreign_decl("C", local, "
#include <stdlib.h>
#include <string.h>
").

    % A string on the heap, which can be overwritten, unlike a literal.
:- pred heap_string(string::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    heap_string(S::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    char *s = malloc(6);
    memcpy(s, \"hello\", 6);
    S = s;
").

:- pred overwrite_string(string::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    overwrite_string(S::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    ((char *) S)[0] = 'J';
").

    % An array of integers: its size, then its elements, a word each.
:- pred overwrite_array(array(int)::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    overwrite_array(A::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    ((MR_Integer *) A)[1] = 99;
").

    % A cell of the only constructor, whose tag is 0.
:- pred overwrite_cell(cell::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    overwrite_cell(C::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    ((MR_Integer *) C)[0] = 99;
").

:- pred check(string::in, bool::in, io::di, io::uo) is det.
check(What, Ok, !IO) :-
    io.write_string(What, !IO),
    io.write_string(( if Ok = yes then ": copied\n" else ": shared\n" ), !IO).

:- pred check_no_sharing(io::di, io::uo) is det.
check_no_sharing(!IO) :-
    heap_string(S, !IO),
    new_mailbox(Strings, !IO),
    send(Strings, S, !IO),
    overwrite_string(S, !IO),
    receive(Strings, any, after(0), RS, !IO),
    check("string", ( if RS = yes("hello") then yes else no ), !IO),

    A = array([1, 2, 3]),
    new_mailbox(Arrays, !IO),
    send(Arrays, A, !IO),
    overwrite_array(A, !IO),
    receive(Arrays, any, after(0), RA, !IO),
    check("array", ( if RA = yes(array([1, 2, 3])) then yes else no ), !IO),

    % Cells made at run time, not static ones, inside a list, a tuple
    % and an array.
    N = array.size(A),
    C1 = cell(N, N),
    C2 = cell(N + 1, N),
    C3 = cell(N + 2, N),
    new_mailbox(Cells, !IO),
    send(Cells, {[C1], array([C2]), C3}, !IO),
    overwrite_cell(C1, !IO),
    overwrite_cell(C2, !IO),
    overwrite_cell(C3, !IO),
    receive(Cells, any, after(0), RC, !IO),
    check("cells in a list, an array and a tuple",
        ( if RC = yes({[cell(3, 3)], array([cell(4, 3)]), cell(5, 3)}) then
            yes
        else
            no
        ), !IO),

    % A message received is a copy too: sending it again and overwriting
    % the received one leaves the second message as it was.
    ( if RA = yes(A1) then
        send(Arrays, A1, !IO),
        overwrite_array(A1, !IO),
        receive(Arrays, any, after(0), RA2, !IO),
        check("a received array sent again",
            ( if RA2 = yes(array([1, 2, 3])) then yes else no ), !IO)
    else
        check("a received array sent again", no, !IO)
    ).

%---------------------------------------------------------------------------%
% The model: each mailbox the program has made, its handle, whether it is
% open, and its queue, oldest first. Stale handles are those of closed
% mailboxes, whose sends must reach no mailbox, even one made later in
% the same slot.

:- type entry ---> entry(mailbox(int), bool, list(int)).

:- type stats
    --->    stats(
                sends :: int,
                stale_sends :: int,
                got :: int,
                got_selected :: int,
                none :: int,
                closes :: int,
                opens :: int,
                bad :: int
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

    % The first element divisible by D, and the rest in order.
:- pred take_first(int::in, list(int)::in, int::out, list(int)::out)
    is semidet.
take_first(D, [X | Xs], Y, Rest) :-
    ( if X mod D = 0 then
        Y = X,
        Rest = Xs
    else
        take_first(D, Xs, Y, Rest0),
        Rest = [X | Rest0]
    ).

:- pred step(int::in, list(entry)::in, list(entry)::out,
    list(mailbox(int))::in, list(mailbox(int))::out, stats::in, stats::out,
    seed::in, seed::out, io::di, io::uo) is det.
step(Step, !Model, !Stale, !Stats, !S, !IO) :-
    next(12, Op, !S),
    next(length(!.Model), I, !S),
    entry(Box, Open, Queue) = det_index0(!.Model, I),
    ( if Op =< 3 then
        next(100, V, !S),
        send(Box, V, !IO),
        ( if Open = yes then
            replace(I, entry(Box, Open, Queue ++ [V]), !Model)
        else
            true
        ),
        !Stats ^ sends := !.Stats ^ sends + 1
    else if Op =< 5 then
        receive(Box, any, after(0), Got, !IO),
        ( if Queue = [X | Rest] then
            Expected = yes(X),
            replace(I, entry(Box, Open, Rest), !Model),
            !Stats ^ got := !.Stats ^ got + 1
        else
            Expected = no,
            !Stats ^ none := !.Stats ^ none + 1
        ),
        compare_step(Step, "receive", Expected, Got, !Stats, !IO)
    else if Op =< 8 then
        next(4, D0, !S),
        D = D0 + 2,
        receive(Box,
            (pred(M::in, M::out) is semidet :- M mod D = 0),
            after(0), Got, !IO),
        ( if take_first(D, Queue, X, Rest) then
            Expected = yes(X),
            replace(I, entry(Box, Open, Rest), !Model),
            !Stats ^ got_selected := !.Stats ^ got_selected + 1
        else
            Expected = no,
            !Stats ^ none := !.Stats ^ none + 1
        ),
        compare_step(Step, "selective receive", Expected, Got, !Stats, !IO)
    else if Op = 9 then
        close_mailbox(Box, !IO),
        ( if Open = yes then
            replace(I, entry(Box, no, []), !Model),
            !:Stale = [Box | !.Stale],
            !Stats ^ closes := !.Stats ^ closes + 1
        else
            true
        )
    else if Op = 10 then
        ( if Open = no then
            new_mailbox(New, !IO),
            replace(I, entry(New, yes, []), !Model),
            !Stats ^ opens := !.Stats ^ opens + 1
        else
            true
        )
    else
        ( if !.Stale = [_ | _] then
            next(length(!.Stale), J, !S),
            next(100, V, !S),
            send(det_index0(!.Stale, J), V, !IO),
            !Stats ^ stale_sends := !.Stats ^ stale_sends + 1
        else
            true
        )
    ).

:- pred compare_step(int::in, string::in, maybe(int)::in, maybe(int)::in,
    stats::in, stats::out, io::di, io::uo) is det.
compare_step(Step, What, Expected, Got, !Stats, !IO) :-
    ( if Expected = Got then
        true
    else
        !Stats ^ bad := !.Stats ^ bad + 1,
        ( if !.Stats ^ bad =< 5 then
            io.write_string("step ", !IO),
            io.write_int(Step, !IO),
            io.write_string(": ", !IO),
            io.write_string(What, !IO),
            io.write_string(" expected ", !IO),
            io.write(Expected, !IO),
            io.write_string(", got ", !IO),
            io.write(Got, !IO),
            io.nl(!IO)
        else
            true
        )
    ).

:- pred steps(int::in, int::in, list(entry)::in, list(entry)::out,
    list(mailbox(int))::in, list(mailbox(int))::out, stats::in, stats::out,
    seed::in, seed::out, io::di, io::uo) is det.
steps(Step, N, !Model, !Stale, !Stats, !S, !IO) :-
    ( if Step >= N then
        true
    else
        step(Step, !Model, !Stale, !Stats, !S, !IO),
        steps(Step + 1, N, !Model, !Stale, !Stats, !S, !IO)
    ).

    % At the end, each open mailbox must hold exactly its model queue.
:- pred drain(list(entry)::in, int::in, int::out, io::di, io::uo) is det.
drain([], !Bad, !IO).
drain([entry(Box, _, Queue) | Es], !Bad, !IO) :-
    check_each(Box, Queue, !Bad, !IO),
    drain(Es, !Bad, !IO).

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
    new_entries(6, Model0, !IO),
    Steps = 40000,
    steps(0, Steps, Model0, Model, [], _, stats(0, 0, 0, 0, 0, 0, 0, 0),
        Stats, 2024, _, !IO),
    drain(Model, 0, Left, !IO),
    io.write_string("model: ", !IO),
    io.write_int(Steps, !IO),
    io.write_string(" steps, ", !IO),
    io.write_int(Stats ^ bad, !IO),
    io.write_string(" differ, ", !IO),
    io.write_int(Left, !IO),
    io.write_string(" queues differ at the end\n", !IO),
    % How much of each kind of step it took, so that a change to the
    % generator that stops covering one shows.
    io.write(Stats ^ bad := 0, !IO),
    io.nl(!IO).

main(!IO) :-
    check_round_trips(!IO),
    check_no_sharing(!IO),
    check_model(!IO).
