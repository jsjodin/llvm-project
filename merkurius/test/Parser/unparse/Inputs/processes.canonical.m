% Process declarations laid out carelessly, for -fprocess-declarations;
% their canonical layout is processes.canonical.m.
:- module processes_messy.
:- interface.
:- import_module process.
:- process counter(int) receives counter_msg.
:- process logger receives list(T).
:- process server(io.res(int), string) receives control_msg,
    (pred(int)),
    {int, data_msg}.
:- process (summer(int) receives sum_msg) on gpu.
:- process(adder(int) receives sum_msg on gpu).
:- implementation.
:- type turn
    --->    on
    ;       off.
p(X) :-
    process,
    X = on,
    process.spawn(q, _, !IO).
