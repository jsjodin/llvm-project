% Closures in messages: a copy carries what the closure captured, by its
% layout, however the capture's type is known (design document, Closures
% that say what they captured).
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_send_closure.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process.

:- typeclass sized(T) where [ func size(T) = int ].
:- instance sized(int) where [ size(X) = X ].
:- instance sized(list(T)) <= sized(T) where [
    size([]) = 0,
    size([X | Xs]) = size(X) + size(Xs)
].

    % A named function, with the curried argument's type in a type variable.
:- func count_plus(list(T), int) = int.
count_plus(L, N) = list.length(L) + N.

    % A lambda expression in polymorphic code, capturing a T.
:- func count_of(T) = (func(int) = int).
count_of(X) = (func(N) = list.length([X, X]) + N).

    % One whose type_info is in a dictionary, made at run time.
:- func sized_of(T) = (func(int) = int) <= sized(T).
sized_of(X) = (func(N) = size([X, X]) + N).

    % A closure that captured a closure.
:- func twice(func(int) = int) = (func(int) = int).
twice(F) = (func(N) = apply(F, apply(F, N))).

    % Lambda expressions whose captured value's type a goal gave.
:- type box
    --->    some [T] box(T) => sized(T)
    ;       some [T] raw(T).

:- func of(box) = (func(int) = int).
of(box(X)) = (func(N) = size(X) + N).
of(raw(X)) = (func(N) = list.length([X]) + N).

:- type holder
    --->    holder(string, func(int) = int).

:- pred child(mailbox(func(int) = int)::in, mailbox(holder)::in,
    mailbox(int)::in, io::di, io::uo) is det.
child(Out, Held, _, !IO) :-
    Base = 100,
    send(Out, (func(N) = N + Base), !IO),
    send(Out, count_plus(["a", "b", "c"]), !IO),
    send(Out, count_of([1.5]), !IO),
    send(Out, sized_of([[1, 2], [3]]), !IO),
    send(Out, twice(count_plus([0])), !IO),
    send(Out, of('new box'(41)), !IO),
    send(Out, of('new raw'("s")), !IO),
    send(Held, holder("held", (func(N) = N * Base)), !IO).

:- pred take(mailbox(func(int) = int)::in, int::in, io::di, io::uo) is det.
take(Out, N, !IO) :-
    ( if N = 0 then
        true
    else
        receive(Out, any, after(20000), Got, !IO),
        (
            Got = yes(F),
            io.write_int(apply(F, 1), !IO),
            io.nl(!IO)
        ;
            Got = no,
            io.write_string("nothing\n", !IO)
        ),
        take(Out, N - 1, !IO)
    ).

main(!IO) :-
    new_mailbox(Out, !IO),
    new_mailbox(Held, !IO),
    spawn(child(Out, Held), _, !IO),
    take(Out, 7, !IO),
    receive(Held, any, after(20000), Got, !IO),
    (
        Got = yes(holder(Name, F)),
        io.write_string(Name, !IO),
        io.write_string(" ", !IO),
        io.write_int(apply(F, 2), !IO),
        io.nl(!IO)
    ;
        Got = no,
        io.write_string("nothing\n", !IO)
    ).

% OUTPUT:101
% OUTPUT-NEXT:4
% OUTPUT-NEXT:3
% OUTPUT-NEXT:13
% OUTPUT-NEXT:3
% OUTPUT-NEXT:42
% OUTPUT-NEXT:2
% OUTPUT-NEXT:held 200
