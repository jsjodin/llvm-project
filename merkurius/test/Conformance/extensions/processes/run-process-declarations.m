% DESIGN: Processes and mailboxes / Process declarations as implemented
% A process declaration, :- process Kind(Types) receives Msg, declares the
% kind's predicate, with the mailbox and io after the arguments.
% spawn(Kind(Args), Box, !IO) spawns the kind, as the kind applied to its
% arguments is a closure of its predicate, and spawn(host, Kind(Args),
% Box, !IO) spawns it at the host; a closure of the kind in a variable
% goes to the library's spawn/5, which runs it at the host. A kind may be
% polymorphic.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/process_declarations_translation.erl
%
% RUN: merkc -O0 -fprocess-declarations %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[1, 2]
% OUTPUT-NEXT:["start", "more"]
% OUTPUT-NEXT:[1.5, 2.5]
% OUTPUT-NOT:{{.}}
:- module run_process_declarations.
:- interface.
:- import_module io, list, process.
:- pred main(io::di, io::uo) is det.

:- type acc_msg(T)
    --->    add(T)
    ;       get(mailbox(list(T))).

:- process accumulator(list(T)) receives acc_msg(T).

:- implementation.

accumulator(Items, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(add(X)),
        accumulator([X | Items], Self, !IO)
    ;
        R = message(get(ReplyTo)),
        send(ReplyTo, list.reverse(Items), !IO)
    ;
        R = timeout
    ).

:- pred ask(mailbox(acc_msg(T))::in, io::di, io::uo) is det.

ask(Box, !IO) :-
    new_mailbox(Back, !IO),
    send(Box, get(Back), !IO),
    receive(Back, any, infinity, R, !IO),
    (
        R = message(Items),
        io.write_line(Items, !IO)
    ;
        R = timeout,
        io.write_string("timeout\n", !IO)
    ).

main(!IO) :-
    spawn(accumulator([]), Ints, !IO),
    send(Ints, add(1), !IO),
    send(Ints, add(2), !IO),
    ask(Ints, !IO),
    spawn(host, accumulator(["start"]), Strings, !IO),
    send(Strings, add("more"), !IO),
    ask(Strings, !IO),
    Kind = accumulator([1.5]),
    spawn(host, Kind, Floats, !IO),
    send(Floats, add(2.5), !IO),
    ask(Floats, !IO).
