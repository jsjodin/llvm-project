% DESIGN: Processes and mailboxes / Process declarations as implemented
% DESIGN: Sized floating point types
% A process declaration, :- process Kind(Types) receives Msg, declares the
% kind's predicate, with the mailbox and io after the arguments.
% spawn(Kind(Args), Box, !IO) spawns the kind, as the kind applied to its
% arguments is a closure of its predicate, and spawn(host, Kind(Args),
% Box, !IO) spawns it at the host; a closure of the kind in a variable
% goes to the library's spawn/5, which runs it at the host. A kind may be
% polymorphic. It is a template, run for float and float32.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/process_declarations_translation.erl
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fprocess-declarations %t.float.d/run_process_declarations_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %t.float.d/run_process_declarations_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:[1, 2]
% OUTPUT-FLOAT-NEXT:["start", "more"]
% OUTPUT-FLOAT-NEXT:[1.5, 2.5]
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fprocess-declarations %t.float32.d/run_process_declarations_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %t.float32.d/run_process_declarations_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:[1, 2]
% OUTPUT-FLOAT32-NEXT:["start", "more"]
% OUTPUT-FLOAT32-NEXT:[1.5f32, 2.5f32]
% OUTPUT-FLOAT32-NOT:{{.}}
:- module run_process_declarations.
:- interface.
:- import_module io, list, process.
:- pred main(io::di, io::uo) is det.

:- type acc_msg(T)
    --->    add(T)
    ;       get(mailbox(list(T))).

:- process accumulator(list(T)) receives acc_msg(T).

:- implementation.
:- import_module @TM@.

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
    Kind = accumulator([1.5@S@]),
    spawn(host, Kind, Floats, !IO),
    send(Floats, add(2.5@S@), !IO),
    ask(Floats, !IO).
