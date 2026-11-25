% DESIGN: Processes and mailboxes / Process declarations as implemented
% receives may list several message types, each a mailbox of its own,
% given to the kind after its arguments in order, so that a client can be
% given only the mailbox it needs. Not yet implemented.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/several_mailboxes_translation.erl
%
% RUN: not merkc -fprocess-declarations %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: process kinds that receive several message types
:- module nyi_several_mailboxes.
:- interface.
:- import_module io, process.
:- pred main(io::di, io::uo) is det.

:- type control
    --->    stop(mailbox(int)).
:- type data
    --->    data(int).

:- process summer(int) receives control, data.

:- implementation.
:- import_module int.

:- type input
    --->    ctl(control)
    ;       add(int).

:- func from_data(data) = input.

from_data(data(N)) = add(N).

    % The oldest message of either mailbox: the data sent before the stop
    % is added first.
summer(Sum, Control, Data, !IO) :-
    Sel = selecting(Control, (func(C) = ctl(C)),
        selecting(Data, from_data, new_selector)),
    receive(Sel, infinity, R, !IO),
    (
        R = message(ctl(stop(ReplyTo))),
        send(ReplyTo, Sum, !IO)
    ;
        R = message(add(N)),
        summer(Sum + N, Control, Data, !IO)
    ;
        R = timeout
    ).

main(!IO) :-
    spawn(summer(0), Control, Data, !IO),
    send(Data, data(1), !IO),
    send(Data, data(2), !IO),
    new_mailbox(Back, !IO),
    send(Control, stop(Back), !IO),
    receive(Back, any, infinity, R, !IO),
    io.write_line(R, !IO).
