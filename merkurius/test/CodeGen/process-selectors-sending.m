% A receive over a selector takes the oldest message while another thread
% sends. A C thread, which no Mercury program can start yet, sends 1, 2,
% 3, ... to two mailboxes in turn while main receives over a selector of
% both, so it must see them in order. The cases look at the mailboxes one
% at a time, so 1 can arrive in the first after it was looked at, and 2 in
% the second before it is: the selector found 2 then, and took it before 1,
% until it learned to look again at a message that arrived as it looked
% (library/process.m, select).
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:received 100000, out of order 0

:- module process_selectors_sending.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process, string.

:- func count = int.
count = 100000.

:- func same(int) = int.
same(N) = N.

main(!IO) :-
    new_mailbox(A : mailbox(int), !IO),
    new_mailbox(B, !IO),
    start_sender(A, B, count, !IO),
    Selector = selecting(A, same, selecting(B, same, new_selector)),
    receive_all(Selector, 1, 0, Received, 0, OutOfOrder, !IO),
    join_sender(!IO),
    io.write_string("received " ++ string.int_to_string(Received) ++
        ", out of order " ++ string.int_to_string(OutOfOrder) ++ "\n", !IO).

:- pred receive_all(selector(int)::in, int::in, int::in, int::out,
    int::in, int::out, io::di, io::uo) is det.

receive_all(Selector, Expected, !Received, !OutOfOrder, !IO) :-
    ( if Expected > count then
        true
    else
        receive(Selector, after(10000), Got, !IO),
        (
            Got = message(N),
            !:Received = !.Received + 1,
            ( if N = Expected then
                true
            else
                !:OutOfOrder = !.OutOfOrder + 1
            ),
            receive_all(Selector, Expected + 1, !Received, !OutOfOrder, !IO)
        ;
            Got = timeout
        )
    ).

:- pragma foreign_decl("C", local, "
#include <pthread.h>
#include <stdint.h>

extern void merk_mailbox_send(MR_Integer node_high, MR_Integer node_low,
    MR_Integer place_slot, MR_Integer generation, MR_Integer secret,
    MR_Word type_info, MR_Word message);

/* A mailbox handle is a cell of five words (library/process.m). */
typedef struct {
    MR_Word *a, *b;
    MR_Word type_info;
    MR_Integer count;
} sender_args;

static pthread_t sender;
static MR_Word handles[2][5];
static sender_args args;

static void *send_alternately(void *raw)
{
    sender_args *s = raw;
    for (MR_Integer i = 1; i <= s->count; ++i) {
        MR_Word *h = i % 2 ? s->a : s->b;
        merk_mailbox_send(h[0], h[1], h[2], h[3], h[4], s->type_info,
            (MR_Word) i);
    }
    return NULL;
}
").

    % Starts the thread that sends 1 to Count to A and B in turn. T is int;
    % the predicate is polymorphic for T's type_info.
:- pred start_sender(mailbox(T)::in, mailbox(T)::in, int::in,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    start_sender(A::in, B::in, Count::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    // The words, as a collection moves the cells.
    for (int i = 0; i < 5; ++i) {
        handles[0][i] = ((MR_Word *) A)[i];
        handles[1][i] = ((MR_Word *) B)[i];
    }
    args.a = handles[0];
    args.b = handles[1];
    args.type_info = TypeInfo_for_T;
    args.count = Count;
    pthread_create(&sender, NULL, send_alternately, &args);
").

:- pred join_sender(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    join_sender(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    pthread_join(sender, NULL);
").
