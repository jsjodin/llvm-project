% A process kind that runs on a GPU and makes a closure, for
% process-kernel-errors.m.
:- module process_kernel_closure.
:- interface.
:- import_module io, process.
:- process adder(int) receives int on gpu.
:- implementation.
:- import_module int.

adder(N, Self, !IO) :-
    Add = (pred(X::in, Y::out) is det :- Y = X + N),
    receive(Self, any, infinity, R, !IO),
    ( if R = message(M) then
        Add(M, Sum),
        adder(Sum, Self, !IO)
    else
        true
    ).
