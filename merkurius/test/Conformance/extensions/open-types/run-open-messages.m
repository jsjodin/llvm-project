% DESIGN: Open terms (Prolog-style variables) / Checking rules
% Open values are sendable with the semantics of Prolog's copy_term: the
% receiver gets one fresh variable for each distinct unbound variable in
% the message, so sharing among them is kept; bound variables arrive as
% their values; and later bindings on either side stay on that side.
% open.send sends a message with holes, and open.receive takes the oldest.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/open_messages_translation.pl
%
% RUN: merkc -O0 -fopen-types %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:received: [1, 7, 1, _1]
% OUTPUT-NEXT:sender's: [_1, 7, _1, 9]
% OUTPUT-NEXT:copy: [3, 3]
% OUTPUT-NEXT:original: [_1, _1]
% OUTPUT-NOT:{{.}}
:- module run_open_messages.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, process, string.

:- type msg
    --->    fill(open(list(int)), mailbox(open(list(int)))).

:- impure pred show(string::in, open(T)::ia, io::di, io::uo) is det.

show(Label, X, !IO) :-
    impure S = open.string(X),
    io.format("%s: %s\n", [s(Label), s(S)], !IO).

    % Binds X to N, if it is unbound: impure, so that it stays between the
    % goals around it, which observe it.
:- impure pred bind(open(int)::ia, int::in) is det.

bind(X, N) :-
    promise_impure ( if X = to_open(N) then true else true ).

    % Binds the first element of the list it gets, and sends it back.
:- pred filler(mailbox(msg)::in, io::di, io::uo) is det.

filler(Self, !IO) :-
    open.receive(Self, infinity, R, !IO),
    promise_pure (
        ( if
            R = message(fill(L, ReplyTo)),
            L = [First | _],
            First = to_open(1)
        then
            impure open.send(ReplyTo, L, !IO)
        else
            true
        )
    ).

main(!IO) :-
    promise_pure (
        L = [A, B, A, C] : open(list(int)),
        impure bind(B, 7),
        new_mailbox(Back, !IO),
        spawn(filler, F, !IO),
        impure open.send(F, fill(L, Back), !IO),
        % Bound after the send: the copy keeps C unbound.
        impure bind(C, 9),
        open.receive(Back, infinity, R, !IO),
        (
            R = message(Got),
            impure show("received", Got, !IO)
        ;
            R = timeout,
            io.write_string("timeout\n", !IO)
        ),
        % The receiver's binding of A's copy does not reach A.
        impure show("sender's", L, !IO),
        % A message to the process's own mailbox is copied too.
        new_mailbox(Own, !IO),
        M = [X, X] : open(list(int)),
        impure open.send(Own, M, !IO),
        open.receive(Own, infinity, R2, !IO),
        ( if R2 = message(Copy), Copy = [Y, _] then
            impure bind(Y, 3),
            impure show("copy", Copy, !IO),
            impure show("original", M, !IO)
        else
            io.write_string("no copy\n", !IO)
        )
    ).
