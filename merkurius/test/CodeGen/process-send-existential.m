% A value of an existentially typed constructor in a message: the type_infos
% and typeclass_infos in its cell are copied with it, including ones made at
% run time (docs/INPROGRESS.md, Closures that say what they captured).
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_send_existential.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process, string.

:- typeclass named(T) where [ func name(T) = string ].
:- typeclass tagged(T) <= named(T) where [ func tag(T) = int ].

:- instance named(int) where [ name(_) = "int" ].
:- instance tagged(int) where [ tag(X) = X ].
:- instance named(list(T)) <= named(T) where [
    name([]) = "empty",
    name([X | _]) = "list of " ++ name(X)
].
:- instance tagged(list(T)) <= tagged(T) where [
    tag(L) = list.length(L)
].

    % A plain type_info, and a dictionary.
:- type box
    --->    some [T] plain(T)
    ;       some [T] classy(T, box) => tagged(T)
    ;       stop.

    % A polymorphic caller makes a type_info and a dictionary at run time.
:- func wrap(T, box) = box <= tagged(T).
wrap(X, Rest) = 'new classy'([X, X], Rest).

:- func plain_of(T) = box.
plain_of(X) = 'new plain'([X]).

:- pred show(box::in, io::di, io::uo) is det.
show(plain(X), !IO) :-
    io.write(X, !IO),
    io.nl(!IO).
show(classy(X, Rest), !IO) :-
    io.write_string(name(X) ++ " ", !IO),
    io.write_int(tag(X), !IO),
    io.nl(!IO),
    show(Rest, !IO).
show(stop, !IO) :-
    io.write_string("stop\n", !IO).

:- pred child(mailbox(box)::in, mailbox(int)::in, io::di, io::uo) is det.
child(Out, _, !IO) :-
    send(Out, wrap(7, wrap([1, 2], plain_of("text"))), !IO),
    send(Out, plain_of(plain_of(1.5)), !IO).

main(!IO) :-
    new_mailbox(Out, !IO),
    spawn(child(Out), _, !IO),
    receive(Out, any, after(20000), A, !IO),
    receive(Out, any, after(20000), B, !IO),
    (
        A = yes(BoxA),
        show(BoxA, !IO)
    ;
        A = no,
        io.write_string("no first\n", !IO)
    ),
    (
        B = yes(_),
        io.write_string("second\n", !IO)
    ;
        B = no,
        io.write_string("no second\n", !IO)
    ).

% OUTPUT:list of int 2
% OUTPUT-NEXT:list of list of int 2
% OUTPUT-NEXT:["text"]
% OUTPUT-NEXT:second
