% Processes and mailboxes, a Merkurius extension with no counterpart in the
% Mercury library (design document, Language extensions, "Processes and
% mailboxes"). So far only the mailboxes of one process: a mailbox is
% made, sent to and received from, with the one-mailbox receive.
%
% A send copies the message, by its type_info, into a fragment of its own
% (runtime/mailbox.cpp), so the sender's cells are never shared with the
% receiver. A receive takes the oldest message its selector accepts and
% leaves the others queued in order, waiting until the timeout for one to
% arrive.
%
% Temporary (chunk 2, docs/INPROGRESS.md): spawn, close_mailbox's effect on
% other processes, selectors over several mailboxes, monitors and request
% and reply come with processes; until then a program is one process, the
% handle has no node or place, and a receive that waits with infinity for
% a message nobody can send waits forever.

:- module process.
:- interface.
:- import_module io.
:- import_module maybe.

    % A mailbox that accepts messages of type M.
:- type mailbox(M).

    % How long a receive waits: milliseconds, after(0) to poll.
:- type timeout
    --->    after(int)
    ;       infinity.

:- pred new_mailbox(mailbox(M)::out, io::di, io::uo) is det.

    % Drops the queued messages; later sends to the mailbox are dropped,
    % and a receive from it finds nothing.
:- pred close_mailbox(mailbox(M)::in, io::di, io::uo) is det.

:- pred send(mailbox(M)::in, M::in, io::di, io::uo) is det.

    % The oldest message for which the selector succeeds, mapped by it, or
    % no if none arrives before the timeout.
:- pred receive(mailbox(M)::in, pred(M, R)::in(pred(in, out) is semidet),
    timeout::in, maybe(R)::out, io::di, io::uo) is det.

    % The selector that takes any message. It never fails, but is declared
    % semidet to fit receive's mode.
:- pred any(M::in, M::out) is semidet.

:- implementation.
:- import_module int.

    % The mailbox's slot in the runtime's table, and the generation that
    % slot had when the mailbox was made, which tells a closed mailbox's
    % handle from that of a later mailbox in the same slot.
:- type mailbox(M)
    --->    mailbox(int, int).

:- pragma foreign_decl("C", local, "
extern void merk_mailbox_new(MR_Integer *slot, MR_Integer *generation);
extern void merk_mailbox_close(MR_Integer slot, MR_Integer generation);
extern void merk_mailbox_send(MR_Integer slot, MR_Integer generation,
    MR_Word type_info, MR_Word message);
extern MR_Integer merk_mailbox_deadline(MR_Integer milliseconds);
extern MR_Integer merk_mailbox_wait(MR_Integer slot, MR_Integer generation,
    MR_Integer position, MR_Integer deadline);
extern MR_Word merk_mailbox_message(MR_Integer slot, MR_Integer generation,
    MR_Integer position);
extern void merk_mailbox_remove(MR_Integer slot, MR_Integer generation,
    MR_Integer position);
").

new_mailbox(mailbox(Slot, Generation), !IO) :-
    new_slot(Slot, Generation, !IO).

close_mailbox(mailbox(Slot, Generation), !IO) :-
    close_slot(Slot, Generation, !IO).

send(mailbox(Slot, Generation), Message, !IO) :-
    send_message(Slot, Generation, Message, !IO).

receive(mailbox(Slot, Generation), Selector, Timeout, Result, !IO) :-
    (
        Timeout = infinity,
        Milliseconds = -1
    ;
        Timeout = after(Milliseconds0),
        Milliseconds = max(Milliseconds0, 0)
    ),
    deadline(Milliseconds, Deadline, !IO),
    scan(Slot, Generation, Selector, Deadline, 0, Result, !IO).

:- pragma foreign_proc("C",
    any(M::in, N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = M;
    SUCCESS_INDICATOR = MR_TRUE;
").

    % From the message at Position on, the first the selector accepts,
    % waiting for the next to arrive until the deadline. Only the
    % mailbox's owner removes messages, so the positions of those it has
    % passed stay as they are.
:- pred scan(int::in, int::in, pred(M, R)::in(pred(in, out) is semidet),
    int::in, int::in, maybe(R)::out, io::di, io::uo) is det.

scan(Slot, Generation, Selector, Deadline, Position, Result, !IO) :-
    wait_message(Slot, Generation, Position, Deadline, Arrived, !IO),
    ( if Arrived = 0 then
        Result = no
    else
        message_at(Slot, Generation, Position, Message, !IO),
        ( if Selector(Message, R) then
            remove_message(Slot, Generation, Position, !IO),
            Result = yes(R)
        else
            scan(Slot, Generation, Selector, Deadline, Position + 1, Result,
                !IO)
        )
    ).

:- pred new_slot(int::out, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    new_slot(Slot::out, Generation::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_new(&Slot, &Generation);
").

:- pred close_slot(int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    close_slot(Slot::in, Generation::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_close(Slot, Generation);
").

:- pred send_message(int::in, int::in, M::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    send_message(Slot::in, Generation::in, Message::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_send(Slot, Generation, TypeInfo_for_M, Message);
").

    % The time at which a receive stops waiting, -1 for never.
:- pred deadline(int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    deadline(Milliseconds::in, Deadline::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Deadline = merk_mailbox_deadline(Milliseconds);
").

    % 1 once the mailbox has a message at Position, 0 if the deadline
    % passes first or the mailbox is closed.
:- pred wait_message(int::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    wait_message(Slot::in, Generation::in, Position::in, Deadline::in,
        Arrived::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Arrived = merk_mailbox_wait(Slot, Generation, Position, Deadline);
").

    % The message at Position, which wait_message has found there.
:- pred message_at(int::in, int::in, int::in, M::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    message_at(Slot::in, Generation::in, Position::in, Message::out,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    (void)TypeInfo_for_M;
    Message = merk_mailbox_message(Slot, Generation, Position);
").

:- pred remove_message(int::in, int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    remove_message(Slot::in, Generation::in, Position::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    merk_mailbox_remove(Slot, Generation, Position);
").
