% A stress test for the collector. Processes each keep a large table of
% mixed terms with a checksum for each, replace random entries so that old
% data dies, check what they read, and send entries to each other around a
% ring. Any mismatch aborts. STRESS_SECONDS, STRESS_WORKERS and
% STRESS_ENTRIES set how long and how large; STRESS_SEED varies the data.
:- module gc_stress.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, char, float, int, list, maybe, pair, process, solutions, string.

:- type tree(T)
    --->    leaf
    ;       node(tree(T), string, T, tree(T)).

:- typeclass score(T) where [ func score(T) = int ].
:- instance score(int) where [ func(score/1) is id_score ].
:- instance score(string) where [ func(score/1) is string.length ].
:- instance score(list(T)) where [ func(score/1) is list.length ].

:- func id_score(int) = int.
id_score(X) = X.

:- type box ---> some [T] box(T) => score(T).

:- type poly(A, B)
    --->    one(A)
    ;       two(A, B)
    ;       many(list(poly(A, B))).

:- type payload
    --->    none
    ;       ints(list(int))
    ;       tree(tree(list(int)))
    ;       text(string)
    ;       boxes(list(box))
    ;       closure(func(int) = int)
    ;       lists(array(list(int)))
    ;       poly(poly(string, list(int)))
    ;       floats(list(float))
    ;       pairs(list(pair(string, int))).

:- type entry ---> entry(int, payload).

:- type message
    --->    next(mailbox(message))
    ;       item(entry).

:- type config ---> config(int, int, int, mailbox(string)).

%---------------------------------------------------------------------------%

:- pred now(int::out, io::di, io::uo) is det.
:- pragma foreign_decl("C", "#include <time.h>\n#include <stdlib.h>\n#include <stdio.h>").
:- pragma foreign_proc("C", now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    T = (MR_Integer) time(NULL);
").

:- pred setting(string::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C", setting(Name::in, Default::in, V::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    const char *value = getenv(Name);
    V = value && *value ? (MR_Integer) atoll(value) : Default;
").

:- pred die(string::in, int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C", die(What::in, A::in, B::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    fflush(stdout);
    fprintf(stderr, \"STRESS FAILED: %s: %lld %lld\\n\", What, (long long) A,
        (long long) B);
    abort();
").

:- pragma foreign_code("C", "
__attribute__((noinline)) MR_Integer stress_address(MR_Word w)
{
    __asm__ volatile(\"\");
    return (MR_Integer) w;
}
").
:- pragma foreign_decl("C", "MR_Integer stress_address(MR_Word w);").

:- pred address(list(int)::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C", address(L::in, A::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    A = stress_address((MR_Word) L);
").

%---------------------------------------------------------------------------%

:- func step(int) = int.
step(X) = X * 6364136223846793005 + 1442695040888963407.

:- pred random(int::in, int::out, int::in, int::out) is det.
random(Bound, V, !R) :-
    !:R = step(!.R),
    V = ((!.R >> 33) /\ 0x3fffffff) mod Bound.

:- func mix(int, int) = int.
mix(H, X) = (H * 31 + X) /\ 0xffffffffffff.

:- func sum_ints(list(int), int) = int.
sum_ints([], H) = H.
sum_ints([X | Xs], H) = sum_ints(Xs, mix(H, X)).

:- pred build_ints(int::in, list(int)::out, int::in, int::out) is det.
build_ints(N, L, !R) :-
    build_ints_2(N, [], L, !R).

:- pred build_ints_2(int::in, list(int)::in, list(int)::out,
    int::in, int::out) is det.
build_ints_2(N, L0, L, !R) :-
    ( if N =< 0 then
        L = L0
    else
        random(1000000, X, !R),
        build_ints_2(N - 1, [X | L0], L, !R)
    ).

:- func insert(string, T, tree(T)) = tree(T).
insert(K, V, leaf) = node(leaf, K, V, leaf).
insert(K, V, node(L, K0, V0, R)) = Result :-
    compare(Order, K, K0),
    (
        Order = (<),
        Result = node(insert(K, V, L), K0, V0, R)
    ;
        Order = (>),
        Result = node(L, K0, V0, insert(K, V, R))
    ;
        Order = (=),
        Result = node(L, K, V, R)
    ).

:- func sum_tree(tree(list(int)), int) = int.
sum_tree(leaf, H) = H.
sum_tree(node(L, K, V, R), H) =
    sum_tree(R, sum_ints(V, sum_string(K, sum_tree(L, H)))).

:- func sum_string(string, int) = int.
sum_string(S, H) = string.foldl(add_char, S, H).

:- func add_char(char, int) = int.
add_char(C, H) = mix(H, char.to_int(C)).

:- pred build_tree(int::in, tree(list(int))::in, tree(list(int))::out,
    int::in, int::out) is det.
build_tree(N, !T, !R) :-
    ( if N =< 0 then
        true
    else
        random(1000000, K, !R),
        random(8, Size, !R),
        build_ints(Size, V, !R),
        !:T = insert("k" ++ string.int_to_string(K), V, !.T),
        build_tree(N - 1, !T, !R)
    ).

:- pred build_text(int::in, string::in, string::out, int::in, int::out) is det.
build_text(N, !S, !R) :-
    ( if N =< 0 then
        true
    else
        random(1000000, X, !R),
        !:S = !.S ++ string.int_to_string(X) ++ ",",
        build_text(N - 1, !S, !R)
    ).

:- pred build_boxes(int::in, list(box)::in, list(box)::out,
    int::in, int::out) is det.
build_boxes(N, !L, !R) :-
    ( if N =< 0 then
        true
    else
        random(3, Kind, !R),
        random(1000, X, !R),
        ( if Kind = 0 then
            B = 'new box'(X)
        else if Kind = 1 then
            B = 'new box'(string.int_to_string(X))
        else
            B = 'new box'(1 `..` (X mod 20))
        ),
        build_boxes(N - 1, [B | !.L], !:L, !R)
    ).

:- func sum_boxes(list(box), int) = int.
sum_boxes([], H) = H.
sum_boxes([box(X) | Xs], H) = sum_boxes(Xs, mix(H, score(X))).

:- pred fill_lists(int::in, array(list(int))::array_di,
    array(list(int))::array_uo, int::in, int::out) is det.
fill_lists(I, !A, !R) :-
    ( if I < array.size(!.A) then
        random(6, Size, !R),
        build_ints(Size, L, !R),
        array.set(I, L, !A),
        fill_lists(I + 1, !A, !R)
    else
        true
    ).

:- func sum_lists(array(list(int)), int, int) = int.
sum_lists(A, I, H) = Result :-
    ( if I < array.size(A) then
        array.lookup(A, I, L),
        Result = sum_lists(A, I + 1, sum_ints(L, H))
    else
        Result = H
    ).

:- pred build_poly(int::in, poly(string, list(int))::out,
    int::in, int::out) is det.
build_poly(Depth, P, !R) :-
    random(3, Kind, !R),
    random(1000, X, !R),
    ( if Depth =< 0 ; Kind = 0 then
        P = one(string.int_to_string(X))
    else if Kind = 1 then
        build_ints(X mod 9, L, !R),
        P = two(string.int_to_string(X), L)
    else
        build_poly(Depth - 1, P1, !R),
        build_poly(Depth - 1, P2, !R),
        build_poly(Depth - 1, P3, !R),
        P = many([P1, P2, P3])
    ).

:- func sum_poly(poly(string, list(int)), int) = int.
sum_poly(one(S), H) = sum_string(S, H).
sum_poly(two(S, L), H) = sum_ints(L, sum_string(S, H)).
sum_poly(many(Ps), H) = sum_polys(Ps, H).

:- func sum_polys(list(poly(string, list(int))), int) = int.
sum_polys([], H) = H.
sum_polys([P | Ps], H) = sum_polys(Ps, sum_poly(P, H)).

:- pred build_floats(int::in, list(float)::in, list(float)::out,
    int::in, int::out) is det.
build_floats(N, !L, !R) :-
    ( if N =< 0 then
        true
    else
        random(1000000, X, !R),
        build_floats(N - 1, [float(X) / 8.0 | !.L], !:L, !R)
    ).

:- func sum_floats(list(float), int) = int.
sum_floats([], H) = H.
sum_floats([X | Xs], H) = sum_floats(Xs, mix(H, truncate_to_int(X * 8.0))).

:- pred build_pairs(int::in, list(pair(string, int))::in,
    list(pair(string, int))::out, int::in, int::out) is det.
build_pairs(N, !L, !R) :-
    ( if N =< 0 then
        true
    else
        random(1000000, X, !R),
        build_pairs(N - 1, [string.int_to_string(X) - X | !.L], !:L, !R)
    ).

:- func sum_pairs(list(pair(string, int)), int) = int.
sum_pairs([], H) = H.
sum_pairs([S - X | Ps], H) = sum_pairs(Ps, mix(sum_string(S, H), X)).

:- func apply_captured(list(int), string, int) = int.
apply_captured(L, S, X) = mix(sum_string(S, sum_ints(L, 0)), X).

:- func checksum(payload) = int.
checksum(none) = 0.
checksum(ints(L)) = sum_ints(L, 1).
checksum(tree(T)) = sum_tree(T, 2).
checksum(text(S)) = sum_string(S, 3).
checksum(boxes(L)) = sum_boxes(L, 4).
checksum(closure(F)) = F(1000).
checksum(lists(A)) = sum_lists(A, 0, 6).
checksum(poly(P)) = sum_poly(P, 7).
checksum(floats(L)) = sum_floats(L, 8).
checksum(pairs(L)) = sum_pairs(L, 9).

:- pred build(payload::out, int::in, int::out) is det.
build(P, !R) :-
    random(100, Kind, !R),
    ( if Kind < 20 then
        random(1500, N, !R),
        build_ints(N, L, !R),
        P = ints(L)
    else if Kind < 35 then
        random(150, N, !R),
        build_tree(N, leaf, T, !R),
        P = tree(T)
    else if Kind < 45 then
        random(200, N, !R),
        build_text(N, "", S, !R),
        P = text(S)
    else if Kind < 58 then
        random(300, N, !R),
        build_boxes(N, [], L, !R),
        P = boxes(L)
    else if Kind < 68 then
        random(200, N, !R),
        build_ints(N, Captured, !R),
        random(1000, X, !R),
        Name = "c" ++ string.int_to_string(X),
        P = closure(apply_captured(Captured, Name))
    else if Kind < 78 then
        random(400, N0, !R),
        % Now and then, an array with a block of its own.
        random(40, Big, !R),
        N = ( if Big = 0 then 20000 + N0 else N0 ),
        array.init(N, [], A0),
        fill_lists(0, A0, A, !R),
        P = lists(A)
    else if Kind < 88 then
        build_poly(6, T, !R),
        P = poly(T)
    else if Kind < 94 then
        random(800, N, !R),
        build_floats(N, [], L, !R),
        P = floats(L)
    else
        random(300, N, !R),
        build_pairs(N, [], L, !R),
        P = pairs(L)
    ).

:- pred new_entry(entry::out, int::in, int::out) is det.
new_entry(entry(H, P), !R) :-
    build(P, !R),
    H = checksum(P).

:- pred check(string::in, entry::in, io::di, io::uo) is det.
check(Where, entry(H, P), !IO) :-
    C = checksum(P),
    ( if C = H then
        true
    else
        die(Where, H, C, !IO)
    ).

%---------------------------------------------------------------------------%

    % Frames that hold terms live across allocations, on a deep stack.
:- pred deep(int::in, int::out, int::in, int::out) is det.
deep(N, H, !R) :-
    ( if N =< 0 then
        H = 0
    else
        random(6, Size, !R),
        build_ints(Size, L, !R),
        Before = sum_ints(L, N),
        deep(N - 1, H0, !R),
        ( if sum_ints(L, N) = Before then
            H = mix(H0, Before)
        else
            H = -1
        )
    ).

    % Nondeterministic code, whose environments hold terms.
:- pred sample(list(int)::in, list(int)::out) is nondet.
sample(L, [X, Y]) :-
    list.member(X, [1 | L]),
    list.member(Y, [2, X]).

:- func sample_sum(list(int)) = int.
sample_sum(L) = list.foldl(plus, list.map(sum_two, solutions(sample(L))), 0).

:- func sum_two(list(int)) = int.
sum_two(L) = list.foldl(plus, L, 0).

:- func expected_sample(list(int)) = int.
expected_sample(L) = list.foldl(plus, list.map(sum_two,
    list.sort_and_remove_dups(list.condense(list.map(per_member, [1 | L])))), 0).

:- func per_member(int) = list(list(int)).
per_member(X) = [[X, 2], [X, X]].

%---------------------------------------------------------------------------%

:- pred fill(int::in, array(entry)::array_di, array(entry)::array_uo,
    int::in, int::out) is det.
fill(I, !T, !R) :-
    ( if I < array.size(!.T) then
        new_entry(E, !R),
        array.set(I, E, !T),
        fill(I + 1, !T, !R)
    else
        true
    ).

:- pred check_all(int::in, array(entry)::array_ui, io::di, io::uo) is det.
check_all(I, T, !IO) :-
    ( if I < array.size(T) then
        array.lookup(T, I, E),
        check("table", E, !IO),
        check_all(I + 1, T, !IO)
    else
        true
    ).

:- pred replace(int::in, array(entry)::array_di, array(entry)::array_uo,
    int::in, int::out, io::di, io::uo) is det.
replace(K, !T, !R, !IO) :-
    ( if K =< 0 then
        true
    else
        random(array.size(!.T), I, !R),
        array.lookup(!.T, I, Old),
        check("replaced", Old, !IO),
        new_entry(E, !R),
        array.set(I, E, !T),
        replace(K - 1, !T, !R, !IO)
    ).

:- pred drain(mailbox(message)::in, array(entry)::array_di,
    array(entry)::array_uo, int::in, int::out, int::in, int::out,
    io::di, io::uo) is det.
drain(Box, !T, !R, !Got, !IO) :-
    receive(Box, any, after(0), M, !IO),
    (
        M = no
    ;
        M = yes(next(_)),
        drain(Box, !T, !R, !Got, !IO)
    ;
        M = yes(item(E)),
        check("message", E, !IO),
        random(array.size(!.T), I, !R),
        array.set(I, E, !T),
        !:Got = !.Got + 1,
        drain(Box, !T, !R, !Got, !IO)
    ).

:- pred round(config::in, mailbox(message)::in, mailbox(message)::in,
    int::in, int::in, int::in,
    list(int)::in, array(entry)::array_di, array(entry)::array_uo,
    int::in, int::in, int::in, io::di, io::uo) is det.
round(Config, Box, Next, Number, Moves, Seed, Sentinel, !.T, !:T,
        !.R, Got, Deadline, !IO) :-
    now(Time, !IO),
    Config = config(_, _, Id, Reports),
    ( if Time >= Deadline then
        check_all(0, !.T, !IO),
        send(Reports, string.format(
            "worker %d: %d rounds, at least %d with a collection, %d received, ok",
            [i(Id), i(Number), i(Moves), i(Got)]), !IO)
    else
        % Reached through an array, so that it is a cell of the heap.
        array.init(1, Sentinel, Holder),
        address(Sentinel, Ignored, !IO),
        array.lookup(Holder, 0, Held),
        address(Held, Before0, !IO),
        Before = Before0 + 0 * Ignored,
        K = int.max(1, array.size(!.T) / 64),
        replace(K, !T, !.R, R1, !IO),
        random(array.size(!.T), I, R1, R2),
        array.lookup(!.T, I, Sent),
        send(Next, item(Sent), !IO),
        random(30000, Depth, R2, R3),
        deep(Depth, H, R3, R4),
        ( if H = -1 then
            die("deep", Depth, Number, !IO)
        else
            true
        ),
        build_ints(40, Sample, R4, R5),
        ( if sample_sum(Sample) = expected_sample(Sample) then
            true
        else
            die("solutions", Number, 0, !IO)
        ),
        drain(Box, !T, R5, R6, Got, Got1, !IO),
        ( if Number mod 32 = 31 then
            check_all(0, !.T, !IO)
        else
            true
        ),
        array.lookup(Holder, 0, Held2),
        address(Held2, After, !IO),
        Moved = ( if Before = After then 0 else 1 ),
        ( if Sentinel = [Seed, Number] then
            true
        else
            die("sentinel", Number, Seed, !IO)
        ),
        round(Config, Box, Next, Number + 1, Moves + Moved, Seed,
            [Seed, Number + 1], !T, R6, Got1, Deadline, !IO)
    ).

:- pred worker(config::in, int::in, mailbox(message)::in, io::di, io::uo) is det.
worker(Config, Deadline, Box, !IO) :-
    receive(Box, any, infinity, First, !IO),
    Config = config(Entries, Seed, Id, Reports),
    ( if First = yes(next(Next)) then
        array.init(Entries, entry(0, none), T0),
        fill(0, T0, T1, Seed + Id, R),
        check_all(0, T1, !IO),
        round(Config, Box, Next, 0, 0, Seed, [Seed, 0], T1, _, R, 0, Deadline, !IO)
    else
        send(Reports, "worker: no ring", !IO)
    ).

:- pred start(int::in, int::in, int::in, int::in, int::in, mailbox(string)::in,
    list(mailbox(message))::in, list(mailbox(message))::out,
    io::di, io::uo) is det.
start(Id, Workers, Entries, Seed, Deadline, Reports, !Boxes, !IO) :-
    ( if Id >= Workers then
        true
    else
        Config = config(Entries, Seed, Id, Reports),
        Body = (pred(Box::in, IO0::di, IO::uo) is det :-
            worker(Config, Deadline, Box, IO0, IO)),
        spawn_with_stack(256 * 1024 * 1024, Body, Mailbox, !IO),
        start(Id + 1, Workers, Entries, Seed, Deadline, Reports,
            [Mailbox | !.Boxes], !:Boxes, !IO)
    ).

:- pred link(list(mailbox(message))::in, mailbox(message)::in,
    io::di, io::uo) is det.
link([], _, !IO).
link([B | Bs], First, !IO) :-
    (
        Bs = [],
        send(B, next(First), !IO)
    ;
        Bs = [Next | _],
        send(B, next(Next), !IO),
        link(Bs, First, !IO)
    ).

:- pred reports(int::in, mailbox(string)::in, io::di, io::uo) is det.
reports(N, Box, !IO) :-
    ( if N =< 0 then
        true
    else
        receive(Box, any, infinity, R, !IO),
        (
            R = yes(Text),
            io.write_string(Text, !IO),
            io.nl(!IO)
        ;
            R = no
        ),
        reports(N - 1, Box, !IO)
    ).

main(!IO) :-
    setting("STRESS_SECONDS", 30, Seconds, !IO),
    setting("STRESS_WORKERS", 4, Workers, !IO),
    setting("STRESS_ENTRIES", 2000, Entries, !IO),
    setting("STRESS_SEED", 12345, Seed, !IO),
    now(Time, !IO),
    new_mailbox(Reports, !IO),
    start(0, Workers, Entries, Seed, Time + Seconds, Reports, [], Boxes, !IO),
    (
        Boxes = [],
        io.write_string("no workers\n", !IO)
    ;
        Boxes = [First | _],
        link(Boxes, First, !IO),
        reports(Workers, Reports, !IO),
        io.write_string("done\n", !IO)
    ).
