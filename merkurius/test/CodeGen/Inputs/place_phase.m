% A second module with a kind that runs on a GPU, for process-places.m: its
% processes run in this module's kernel, merk_place, not the other's.
:- module place_phase.
:- interface.
:- import_module io, process.

    % Sends Reply the number it receives, doubled, and ends.
:- process doubler(mailbox(int)) receives int on gpu.

:- implementation.
:- import_module int.

doubler(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        send(Reply, 2 * N, !IO)
    else
        true
    ).
