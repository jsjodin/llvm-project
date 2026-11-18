% process-group-private.m's program under fallback(abort), with an array of
% 20000 ints, which does not fit in a team's local memory
% (process-group-private-gpu.test).

:- module process_group_private_abort.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- process worker(mailbox(int)) receives int on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(T), T > 0 then
        Tiles = 50,
        array.init(Tiles * T, 0, Out0),
        tiles(Tiles, T, Out0, Out),
        ( for(I, 0, Tiles * T - 1), fromto(0, S0, S1, Sum) do
            S1 = S0 + Out ^ unsafe_elem(I)
        ),
        send(Reply, Sum, !IO),
        worker(Reply, Self, !IO)
    else
        true
    ).

    % Each tile's T elements, through a group-private array G: filled in
    % order, and read back reversed.
:- pred tiles(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
tiles(Tiles, T, !Out) :-
    %$omp teams distribute dyn_groupprivate(fallback(abort): G0)
    ( for(J, 0, Tiles - 1) do
        array.init(T, 0, G0),
        fill(J, T, G0, G),
        reverse_into(J, T, G, !Out)
    ).

:- pred fill(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
fill(J, T, !G) :-
    %$omp parallel do
    ( for(K, 0, T - 1) do
        !G ^ unsafe_elem(K) := J * 1000 + K
    ).

:- pred reverse_into(int::in, int::in, array(int)::array_ui,
    array(int)::array_di, array(int)::array_uo) is det.
reverse_into(J, T, G, !Out) :-
    %$omp parallel do
    ( for(K, 0, T - 1) do
        !Out ^ unsafe_elem(J * T + K) := G ^ unsafe_elem(T - 1 - K) * (K + 1)
    ).

main(!IO) :-
    new_mailbox(Me, !IO),
    spawn(gpu(0), worker(Me), W, !IO),
    list.foldl(run(Me, W), [16, 300, 5000, 20000], !IO),
    send(W, 0, !IO).

:- pred run(mailbox(int)::in, mailbox(int)::in, int::in, io::di, io::uo)
    is det.
run(Me, W, T, !IO) :-
    send(W, T, !IO),
    receive(Me, any, after(20000), R, !IO),
    ( if R = message(S) then
        io.format("%d: %d\n", [i(T), i(S)], !IO)
    else
        io.write_string("no answer\n", !IO)
    ).
