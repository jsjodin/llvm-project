% MANUAL: Combined higher-order types and insts
% A ground term's argument whose type is a higher-order type with modes
% and a determinism has that inst when a unification deconstructs it, or a
% field access gets it, and can be called. The manual's job type, a
% function's, one with no arguments, and constructors of several such
% arguments.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:7
% OUTPUT-NEXT:create 7
% OUTPUT-NEXT:delete 8
% OUTPUT-NEXT:7 positive
% OUTPUT-NEXT:-5 not positive
% OUTPUT-NEXT:30
% OUTPUT-NEXT:started
% OUTPUT-NOT:{{.}}

:- module run_manual_example.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type job
    --->    job(pred(int::out, io::di, io::uo) is det).

:- type task
    --->    create(pred(int::in, io::di, io::uo) is det)
    ;       delete(pred(int::in, io::di, io::uo) is det).

:- type calc
    --->    calc(
                op      :: (func(int::in, int::in) = (int::out) is det),
                check   :: (pred(int::in) is semidet),
                start   :: ((pred) is det)
            ).

main(!IO) :-
    run(job(seven), !IO),
    perform(create(show("create")), !IO),
    perform(delete(show("delete")), !IO),
    C = calc(add, positive, nothing),
    apply(C, 3, 4, !IO),
    apply(C, -9, 4, !IO),
    Op = C ^ op,
    io.write_line(Op(10, 20), !IO),
    Start = C ^ start,
    Start,
    io.write_string("started\n", !IO).

:- pred run(job::in, io::di, io::uo) is det.
run(Job, !IO) :-
    Job = job(Pred),
    Pred(Result, !IO),
    io.write_line(Result, !IO).

:- pred perform(task::in, io::di, io::uo) is det.
perform(create(P), !IO) :- P(7, !IO).
perform(delete(P), !IO) :- P(8, !IO).

:- pred apply(calc::in, int::in, int::in, io::di, io::uo) is det.
apply(calc(Op, Check, _), X, Y, !IO) :-
    Z = Op(X, Y),
    ( if Check(Z) then
        io.format("%d positive\n", [i(Z)], !IO)
    else
        io.format("%d not positive\n", [i(Z)], !IO)
    ).

:- pred seven(int::out, io::di, io::uo) is det.
seven(7, !IO).

:- pred show(string::in, int::in, io::di, io::uo) is det.
show(What, N, !IO) :-
    io.format("%s %d\n", [s(What), i(N)], !IO).

:- func add(int, int) = int.
add(X, Y) = X + Y.

:- pred positive(int::in) is semidet.
positive(X) :- X > 0.

:- pred nothing is det.
nothing.
