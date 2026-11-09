% A process's heap and the messages it read are freed when it ends, so that
% many short processes stay within a small amount of memory (design document,
% Memory management). Without that this needs over 2 GB.
%
% RUN: merkc -O2 %s -o %t
% RUN: %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_WORKERS=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_end_frees.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process.

:- pragma foreign_decl("C", local, "#include <sys/resource.h>").

:- pred peak_megabytes(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    peak_megabytes(M::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    M = usage.ru_maxrss / 1024;
").

:- func make(int, list(int)) = list(int).
make(N, L) = ( if N = 0 then L else make(N - 1, [N | L]) ).

:- pred child(mailbox(int)::in, mailbox(list(int))::in, io::di, io::uo) is det.
child(Out, Box, !IO) :-
    Mine = make(20000, []),
    receive(Box, any, after(20000), Got, !IO),
    (
        Got = message(L),
        send(Out, list.length(L) + list.length(Mine), !IO)
    ;
        Got = timeout,
        send(Out, -1, !IO)
    ).

:- pred run(int::in, list(int)::in, mailbox(int)::in, int::in, int::out,
    io::di, io::uo) is det.
run(N, Message, Out, !Sum, !IO) :-
    ( if N = 0 then
        true
    else
        spawn(child(Out), Box, !IO),
        send(Box, Message, !IO),
        receive(Out, any, after(20000), Got, !IO),
        (
            Got = message(V),
            !:Sum = !.Sum + V
        ;
            Got = timeout
        ),
        run(N - 1, Message, Out, !Sum, !IO)
    ).

main(!IO) :-
    new_mailbox(Out, !IO),
    run(3000, make(20000, []), Out, 0, Sum, !IO),
    io.write_int(Sum, !IO),
    io.nl(!IO),
    peak_megabytes(Peak, !IO),
    ( if Peak < 400 then
        io.write_string("within bounds\n", !IO)
    else
        io.write_int(Peak, !IO),
        io.write_string(" MB\n", !IO)
    ).

% OUTPUT:120000000
% OUTPUT-NEXT:within bounds
