% Messages between GPU processes of one kernel, through the receiver's queue
% in the GPU's memory, with no call of the host (docs/INPROGRESS.md,
% Messages between GPU processes on one GPU), beside those from the host,
% which go through the host's mailbox and the ring it fills.
%
% - "order": three senders each send a collector 300 numbered words, more
%   than its queue holds, and then tell the main process; the main process
%   then sends the collector -1, through the host. The collector takes the
%   900 words and the -1 in whatever order they come between the senders,
%   and finds each sender's in the order sent.
% - "arrays": the same with arrays of 1 to 50 elements, each element the
%   sender and the array's number, and two arrays from the host.
% - "terms": three senders each send a collector 100 arrays of 1 to 50
%   elements twice, in a term with the array's number: the first send copies
%   the array, which is sent again, and the second, its last use, moves it
%   (merk.foreign_call's `moves`). The collector checks each array and the
%   order.
% - "self": a process sends itself 200 words before it takes any, so that
%   most of them spill from its queue's slots, and takes them in order.
% - "quit": a process takes one word and ends, while three senders send it
%   300 words each and arrays of 4000 words; what it did not take goes as it
%   ends, and what is sent after it, as to a closed mailbox.
% - "closed": the main process closes a waiter's mailbox, which may be
%   receiving already, before a sender sends it ten words: the waiter's
%   receive is an error, as a receive from a closed mailbox is. At the host
%   the error ends the waiter; on the GPU it ends the program.
%
% The kinds run at the host here, and on the GPU in process-direct-gpu.test,
% where the sends between them are direct.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host order | FileCheck %s --check-prefix=ORDER --match-full-lines --strict-whitespace
% RUN: %libomp-env %t2 host arrays | FileCheck %s --check-prefix=ARRAYS --match-full-lines --strict-whitespace
% RUN: %libomp-env %t2 host terms | FileCheck %s --check-prefix=TERMS --match-full-lines --strict-whitespace
% RUN: %libomp-env %t2 host self | FileCheck %s --check-prefix=SELF --match-full-lines --strict-whitespace
% RUN: %libomp-env %t2 host quit | FileCheck %s --check-prefix=QUIT --match-full-lines --strict-whitespace
% RUN: %libomp-env %t2 host closed 2>&1 | FileCheck %s --check-prefix=CLOSED
% RUN: merkc -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t
% RUN: %libomp-env %t host order | FileCheck %s --check-prefix=ORDER --match-full-lines --strict-whitespace
% RUN: %libomp-env %t host arrays | FileCheck %s --check-prefix=ARRAYS --match-full-lines --strict-whitespace

:- module process_direct.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- type report
    --->    sent(int)
    ;       got(int, int)
    ;       timed_out(int, int).

:- type tagged
    --->    tagged(int, array(int)).

    % Sends To the words S * 1000000 + I for I from 1 to K, and then tells
    % Report.
:- process sender(int, int, mailbox(int), mailbox(report)) receives int
    on gpu.

sender(S, K, To, Report, _, !IO) :-
    send_words(S, 1, K, To, !IO),
    send(Report, sent(S), !IO).

:- pred send_words(int::in, int::in, int::in, mailbox(int)::in,
    io::di, io::uo) is det.

send_words(S, I, K, To, !IO) :-
    ( if I =< K then
        send(To, S * 1000000 + I, !IO),
        send_words(S, I + 1, K, To, !IO)
    else
        true
    ).

    % Takes N words of senders 0 to 2, and -1, and reports how many it took
    % and how many were out of their sender's order.
:- process collector(int, mailbox(report)) receives int on gpu.

collector(N, Report, Self, !IO) :-
    collect(N + 1, 0, 0, 0, 0, 0, Report, Self, !IO).

:- pred collect(int::in, int::in, int::in, int::in, int::in, int::in,
    mailbox(report)::in, mailbox(int)::in, io::di, io::uo) is det.

collect(Left, Took, Wrong, L0, L1, L2, Report, Self, !IO) :-
    ( if Left = 0 then
        send(Report, got(Took, Wrong), !IO)
    else
        receive(Self, any, after(10000), R, !IO),
        ( if R = message(V) then
            ( if V < 0 then
                collect(Left - 1, Took + 1, Wrong, L0, L1, L2, Report, Self,
                    !IO)
            else
                S = V / 1000000,
                I = V mod 1000000,
                ( if S = 0 then
                    W = ( if I = L0 + 1 then Wrong else Wrong + 1 ),
                    collect(Left - 1, Took + 1, W, I, L1, L2, Report, Self,
                        !IO)
                else if S = 1 then
                    W = ( if I = L1 + 1 then Wrong else Wrong + 1 ),
                    collect(Left - 1, Took + 1, W, L0, I, L2, Report, Self,
                        !IO)
                else
                    W = ( if I = L2 + 1 then Wrong else Wrong + 1 ),
                    collect(Left - 1, Took + 1, W, L0, L1, I, Report, Self,
                        !IO)
                )
            )
        else
            send(Report, timed_out(Took, Wrong), !IO)
        )
    ).

    % Sends To K arrays, the I'th of 1 + I mod 50 elements, each
    % S * 1000000 + I, and then tells Report.
:- process array_sender(int, int, mailbox(array(int)), mailbox(report))
    receives int on gpu.

array_sender(S, K, To, Report, _, !IO) :-
    send_arrays(S, 1, K, To, !IO),
    send(Report, sent(S), !IO).

:- pred send_arrays(int::in, int::in, int::in, mailbox(array(int))::in,
    io::di, io::uo) is det.

send_arrays(S, I, K, To, !IO) :-
    ( if I =< K then
        send(To, array.init(1 + I mod 50, S * 1000000 + I), !IO),
        send_arrays(S, I + 1, K, To, !IO)
    else
        true
    ).

    % Sends To K terms tagged(S * 1000000 + I, A) twice, A the I'th array of
    % send_arrays, and then tells Report.
:- process term_sender(int, int, mailbox(tagged), mailbox(report))
    receives int on gpu.

term_sender(S, K, To, Report, _, !IO) :-
    send_terms(S, 1, K, To, !IO),
    send(Report, sent(S), !IO).

:- pred send_terms(int::in, int::in, int::in, mailbox(tagged)::in,
    io::di, io::uo) is det.

send_terms(S, I, K, To, !IO) :-
    ( if I =< K then
        V = S * 1000000 + I,
        A = array.init(1 + I mod 50, V),
        send(To, tagged(V, A), !IO),
        send(To, tagged(V, A), !IO),
        send_terms(S, I + 1, K, To, !IO)
    else
        true
    ).

    % Takes N terms of senders 0 to 2, and reports how many it took and how
    % many were out of their sender's order or not as sent: each sender's
    % numbers come twice each, in order.
:- process term_collector(int, mailbox(report)) receives tagged on gpu.

term_collector(N, Report, Self, !IO) :-
    collect_terms(N, 0, 0, 0, 0, 0, Report, Self, !IO).

:- pred collect_terms(int::in, int::in, int::in, int::in, int::in, int::in,
    mailbox(report)::in, mailbox(tagged)::in, io::di, io::uo) is det.

collect_terms(Left, Took, Wrong, L0, L1, L2, Report, Self, !IO) :-
    ( if Left = 0 then
        send(Report, got(Took, Wrong), !IO)
    else
        receive(Self, any, after(10000), R, !IO),
        ( if R = message(tagged(V, A)) then
            S = V / 1000000,
            I = V mod 1000000,
            Shape = ( if array.size(A) = 1 + I mod 50, all_are(A, V, 0)
                then 0 else 1 ),
            Last = ( if S = 0 then L0 else if S = 1 then L1 else L2 ),
            % Two sends of each I: the second count is I itself.
            Order = ( if I * 2 - 1 = Last + 1 then 0
                else if I * 2 = Last + 1 then 0 else 1 ),
            W = Wrong + Shape + Order,
            ( if S = 0 then
                collect_terms(Left - 1, Took + 1, W, Last + 1, L1, L2, Report,
                    Self, !IO)
            else if S = 1 then
                collect_terms(Left - 1, Took + 1, W, L0, Last + 1, L2, Report,
                    Self, !IO)
            else
                collect_terms(Left - 1, Took + 1, W, L0, L1, Last + 1, Report,
                    Self, !IO)
            )
        else
            send(Report, timed_out(Took, Wrong), !IO)
        )
    ).

    % Takes N arrays of senders 0 to 2 and two of the host's, whose first
    % element is negative, and reports how many it took and how many were
    % out of their sender's order or not as sent.
:- process array_collector(int, mailbox(report)) receives array(int)
    on gpu.

array_collector(N, Report, Self, !IO) :-
    collect_arrays(N + 2, 0, 0, 0, 0, 0, Report, Self, !IO).

:- pred collect_arrays(int::in, int::in, int::in, int::in, int::in, int::in,
    mailbox(report)::in, mailbox(array(int))::in, io::di, io::uo) is det.

collect_arrays(Left, Took, Wrong, L0, L1, L2, Report, Self, !IO) :-
    ( if Left = 0 then
        send(Report, got(Took, Wrong), !IO)
    else
        receive(Self, any, after(10000), R, !IO),
        ( if R = message(A) then
            V = array.lookup(A, 0),
            ( if V < 0 then
                W0 = ( if array.size(A) = 3 then Wrong else Wrong + 1 ),
                collect_arrays(Left - 1, Took + 1, W0, L0, L1, L2, Report,
                    Self, !IO)
            else
                S = V / 1000000,
                I = V mod 1000000,
                Shape = ( if array.size(A) = 1 + I mod 50,
                    all_are(A, V, 0) then 0 else 1 ),
                ( if S = 0 then
                    W = Wrong + Shape + ( if I = L0 + 1 then 0 else 1 ),
                    collect_arrays(Left - 1, Took + 1, W, I, L1, L2, Report,
                        Self, !IO)
                else if S = 1 then
                    W = Wrong + Shape + ( if I = L1 + 1 then 0 else 1 ),
                    collect_arrays(Left - 1, Took + 1, W, L0, I, L2, Report,
                        Self, !IO)
                else
                    W = Wrong + Shape + ( if I = L2 + 1 then 0 else 1 ),
                    collect_arrays(Left - 1, Took + 1, W, L0, L1, I, Report,
                        Self, !IO)
                )
            )
        else
            send(Report, timed_out(Took, Wrong), !IO)
        )
    ).

:- pred all_are(array(int)::in, int::in, int::in) is semidet.

all_are(A, V, I) :-
    ( if I < array.size(A) then
        array.lookup(A, I) = V,
        all_are(A, V, I + 1)
    else
        true
    ).

    % Sends itself K words before it takes any, and reports how many it
    % took and how many were out of order.
:- process selfie(int, mailbox(report)) receives int on gpu.

selfie(K, Report, Self, !IO) :-
    send_words(0, 1, K, Self, !IO),
    collect(K, 0, 0, 0, 0, 0, Report, Self, !IO).

    % Takes one word, reports it, and ends.
:- process quitter(mailbox(report)) receives int on gpu.

quitter(Report, Self, !IO) :-
    receive(Self, any, after(10000), R, !IO),
    ( if R = message(V) then
        send(Report, got(1, V), !IO)
    else
        send(Report, timed_out(0, 0), !IO)
    ).

    % Sends To K arrays of 4000 words, and then tells Report.
:- process big_sender(int, int, mailbox(array(int)), mailbox(report))
    receives int on gpu.

big_sender(S, K, To, Report, _, !IO) :-
    send_big(K, To, !IO),
    send(Report, sent(S), !IO).

:- pred send_big(int::in, mailbox(array(int))::in, io::di, io::uo) is det.

send_big(K, To, !IO) :-
    ( if K > 0 then
        send(To, array.init(4000, K), !IO),
        send_big(K - 1, To, !IO)
    else
        true
    ).

    % Takes one array, reports its size, and ends.
:- process array_quitter(mailbox(report)) receives array(int) on gpu.

array_quitter(Report, Self, !IO) :-
    receive(Self, any, after(10000), R, !IO),
    ( if R = message(A) then
        send(Report, got(1, array.size(A)), !IO)
    else
        send(Report, timed_out(0, 0), !IO)
    ).

    % Takes words until none comes for half a second, and reports how many.
:- process waiter(mailbox(report)) receives int on gpu.

waiter(Report, Self, !IO) :-
    wait_for(0, Report, Self, !IO).

:- pred wait_for(int::in, mailbox(report)::in, mailbox(int)::in,
    io::di, io::uo) is det.

wait_for(N, Report, Self, !IO) :-
    receive(Self, any, after(500), R, !IO),
    ( if R = message(_) then
        wait_for(N + 1, Report, Self, !IO)
    else
        send(Report, timed_out(N, 0), !IO)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = [Where, Case] then
        Place = ( if Where = "gpu" then gpu(0) else host ),
        new_mailbox(Reports, !IO),
        ( if Case = "order" then
            order(Place, Reports, !IO)
        else if Case = "arrays" then
            arrays(Place, Reports, !IO)
        else if Case = "terms" then
            spawn(Place, term_collector(600, Reports), C, !IO),
            spawn(Place, term_sender(0, 100, C, Reports), _, !IO),
            spawn(Place, term_sender(1, 100, C, Reports), _, !IO),
            spawn(Place, term_sender(2, 100, C, Reports), _, !IO),
            print_reports(4, Reports, !IO)
        else if Case = "self" then
            spawn(Place, selfie(200, Reports), _, !IO),
            print_reports(1, Reports, !IO)
        else if Case = "quit" then
            quit(Place, Reports, !IO)
        else if Case = "closed" then
            spawn(Place, waiter(Reports), W, !IO),
            close_mailbox(W, !IO),
            spawn(Place, sender(0, 10, W, Reports), _, !IO),
            receive(Reports, any, after(2000), _, !IO),
            receive(Reports, any, after(2000), _, !IO)
        else
            io.write_string("usage: host or gpu, then a case\n", !IO)
        )
    else
        io.write_string("usage: host or gpu, then a case\n", !IO)
    ).

:- pred order(place::in, mailbox(report)::in, io::di, io::uo) is det.

order(Place, Reports, !IO) :-
    spawn(Place, collector(900, Reports), C, !IO),
    spawn(Place, sender(0, 300, C, Reports), _, !IO),
    spawn(Place, sender(1, 300, C, Reports), _, !IO),
    spawn(Place, sender(2, 300, C, Reports), _, !IO),
    print_reports(3, Reports, !IO),
    send(C, -1, !IO),
    print_reports(1, Reports, !IO).

:- pred arrays(place::in, mailbox(report)::in, io::di, io::uo) is det.

arrays(Place, Reports, !IO) :-
    spawn(Place, array_collector(900, Reports), C, !IO),
    send(C, array([-1, 2, 3]), !IO),
    spawn(Place, array_sender(0, 300, C, Reports), _, !IO),
    spawn(Place, array_sender(1, 300, C, Reports), _, !IO),
    spawn(Place, array_sender(2, 300, C, Reports), _, !IO),
    print_reports(3, Reports, !IO),
    send(C, array([-2, 2, 3]), !IO),
    print_reports(1, Reports, !IO).

:- pred quit(place::in, mailbox(report)::in, io::di, io::uo) is det.

quit(Place, Reports, !IO) :-
    spawn(Place, quitter(Reports), Q, !IO),
    spawn(Place, sender(0, 300, Q, Reports), _, !IO),
    spawn(Place, sender(1, 300, Q, Reports), _, !IO),
    spawn(Place, sender(2, 300, Q, Reports), _, !IO),
    spawn(Place, array_quitter(Reports), A, !IO),
    spawn(Place, big_sender(3, 20, A, Reports), _, !IO),
    spawn(Place, big_sender(4, 20, A, Reports), _, !IO),
    sum_reports(7, 0, 0, 0, Reports, !IO).

    % Prints the reports in a fixed order: the sends sorted, the rest after.
:- pred print_reports(int::in, mailbox(report)::in, io::di, io::uo) is det.

print_reports(N, Reports, !IO) :-
    take_reports(N, Reports, [], Got, !IO),
    list.sort(Got, Sorted),
    list.foldl(print_report, Sorted, !IO).

:- pred take_reports(int::in, mailbox(report)::in, list(report)::in,
    list(report)::out, io::di, io::uo) is det.

take_reports(N, Reports, !Got, !IO) :-
    ( if N > 0 then
        receive(Reports, any, after(20000), R, !IO),
        ( if R = message(Report) then
            !:Got = [Report | !.Got]
        else
            !:Got = [timed_out(-1, -1) | !.Got]
        ),
        take_reports(N - 1, Reports, !Got, !IO)
    else
        true
    ).

:- pred print_report(report::in, io::di, io::uo) is det.

print_report(Report, !IO) :-
    io.write(Report, !IO),
    io.nl(!IO).

    % The quitters' reports, whose words depend on which message came
    % first: how many sends ended, how many quitters took one, and whether
    % each took a message that was sent.
:- pred sum_reports(int::in, int::in, int::in, int::in, mailbox(report)::in,
    io::di, io::uo) is det.

sum_reports(N, Sent, Took, Wrong, Reports, !IO) :-
    ( if N > 0 then
        receive(Reports, any, after(20000), R, !IO),
        ( if R = message(sent(_)) then
            sum_reports(N - 1, Sent + 1, Took, Wrong, Reports, !IO)
        else if R = message(got(_, V)) then
            Ok = ( if V = 4000 ; V mod 1000000 = 1 then 0 else 1 ),
            sum_reports(N - 1, Sent, Took + 1, Wrong + Ok, Reports, !IO)
        else
            sum_reports(N - 1, Sent, Took, Wrong + 1, Reports, !IO)
        )
    else
        io.format("sent %d, took %d, wrong %d\n", [i(Sent), i(Took),
            i(Wrong)], !IO)
    ).
% ORDER:sent(0)
% ORDER-NEXT:sent(1)
% ORDER-NEXT:sent(2)
% ORDER-NEXT:got(901, 0)
% ARRAYS:sent(0)
% ARRAYS-NEXT:sent(1)
% ARRAYS-NEXT:sent(2)
% ARRAYS-NEXT:got(902, 0)
% TERMS:sent(0)
% TERMS-NEXT:sent(1)
% TERMS-NEXT:sent(2)
% TERMS-NEXT:got(600, 0)
% SELF:got(200, 0)
% QUIT:sent 5, took 2, wrong 0
% FREED-NEXT:Mercury runtime: 0 messages are neither read nor freed
% FREED-NEXT:Mercury runtime: 0 chunks of the devices' arenas and 0 rings are allocated
% FREED-NEXT:Mercury runtime: 0 blocks of messages between GPU processes are allocated
% CLOSED:Software Error: receive: the mailbox is closed or another node's
