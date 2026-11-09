% A send copies its message by the message's type_info, here in
% polymorphic code, where the message is a word: a float, a character, a
% tuple, a notag, arrays of strings and of lists, and a list of terms of a
% type whose constructors share a tag, with secondary tags. A changed copy
% of a sent array does not change the message. mmc accepts the program
% with process.m, but has no process module to run it with, so the output
% is written by hand.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_messages.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, char, int, list, maybe, process, string.

:- type many
    --->    none
    ;       a(int) ; b(string) ; c(float) ; d(int, int) ; e(many)
    ;       f(list(int)) ; g(char) ; h({int, string}).

:- type wrap ---> wrap(string).

:- pred show(received(T)::in, io::di, io::uo) is det.
show(M, !IO) :-
    io.write(M, !IO),
    io.nl(!IO).

    % Sends and receives a message in polymorphic code, through its
    % type_info.
:- pred round_trip(T::in, received(T)::out, io::di, io::uo) is det.
round_trip(X, R, !IO) :-
    new_mailbox(Box, !IO),
    send(Box, X, !IO),
    receive(Box, any, after(0), R, !IO).

main(!IO) :-
    round_trip(2.75, F, !IO), show(F, !IO),
    round_trip('λ', C, !IO), show(C, !IO),
    round_trip({1, "two", 3.0}, T, !IO), show(T, !IO),
    round_trip(wrap("notag"), W, !IO), show(W, !IO),
    round_trip(array(["x", "y"]), AS, !IO), show(AS, !IO),
    round_trip(array([[1, 2], [3]]), AL, !IO), show(AL, !IO),
    round_trip([e(h({7, "seven"})), g('q'), f([4, 5]), d(1, 2), c(0.5),
        b("bee"), a(-3), none], M, !IO),
    show(M, !IO),
    % A changed copy of the sent array does not change the message.
    A0 = array([10, 20]),
    new_mailbox(Arrays, !IO),
    send(Arrays, A0, !IO),
    A1 = slow_set(A0, 0, 99),
    receive(Arrays, any, after(0), Got, !IO),
    show(Got, !IO),
    io.write(A1, !IO),
    io.nl(!IO).

% OUTPUT:message(2.75)
% OUTPUT-NEXT:message('λ')
% OUTPUT-NEXT:message({1, "two", 3.0})
% OUTPUT-NEXT:message(wrap("notag"))
% OUTPUT-NEXT:message(array(["x", "y"]))
% OUTPUT-NEXT:message(array({{\[\[}}1, 2], [3]]))
% OUTPUT-NEXT:message([e(h({7, "seven"})), g('q'), f([4, 5]), d(1, 2), c(0.5), b("bee"), a(-3), none])
% OUTPUT-NEXT:message(array([10, 20]))
% OUTPUT-NEXT:array([99, 20])
% OUTPUT-EMPTY:
