% The translation of run-open-messages.m into Prolog. A mailbox is a
% message queue, and a process a thread: thread_send_message copies the
% message as copy_term does, a fresh variable for each distinct unbound
% one, as open.send does.
:- ensure_loaded(mercury_write).

show(Label, X) :-
    with_output_to(string(S), mercury_write(X)),
    format("~w: ~w~n", [Label, S]).

bind(X, N) :-
    ( X = N -> true ; true ).

filler(Self) :-
    thread_get_message(Self, fill(L, ReplyTo)),
    (   L = [First | _], First = 1
    ->  thread_send_message(ReplyTo, L)
    ;   true
    ).

main :-
    L = [A, B, A, C],
    bind(B, 7),
    message_queue_create(Back),
    message_queue_create(F),
    thread_create(filler(F), Thread, []),
    thread_send_message(F, fill(L, Back)),
    bind(C, 9),
    thread_get_message(Back, Got),
    show("received", Got),
    show("sender's", L),
    message_queue_create(Own),
    M = [X, X],
    thread_send_message(Own, M),
    thread_get_message(Own, Copy),
    (   Copy = [Y, _]
    ->  bind(Y, 3),
        show("copy", Copy),
        show("original", M)
    ;   write('no copy\n')
    ),
    thread_join(Thread, _).
