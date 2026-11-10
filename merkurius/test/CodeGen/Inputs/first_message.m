% A module with a receive goal, for extension-modules.m.
:- module first_message.
:- interface.
:- import_module io, process.
    % The mailbox's first message, or -1 if none is queued.
:- pred first_message(mailbox(int)::in, int::out, io::di, io::uo) is det.
:- implementation.
first_message(Mailbox, N, !IO) :-
    receive [timeout(after(0)), io(!IO)] (
        select(Mailbox, M) -> N = M
    ;
        timeout -> N = -1
    ).
