% A barrier between the teams of a target teams region, from team mailboxes
% (README.md; docs/INPROGRESS.md, Processes as teams, step 5). Every team
% arrives, and is released once all have. Two shapes:
%
%   central   each team sends its arrival to team 0, which then releases each
%   tree      a binary tree: a team waits for its children's arrivals, sends
%             its own to its parent, waits for the release, and passes it on
%             to its children
%
% An arrival in round R is the message 2R, a release 2R + 1, and each team
% counts the messages it did not expect. The rounds are timed on the host with
% clock_gettime, less an empty region's time, which holds the launch and the
% teams' wait for each other to start.
%
% Usage: teambarrier central|tree TEAMS ROUNDS

:- module teambarrier.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- func warm_up = int.

warm_up = 100.

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if
        Arguments = [Shape, TeamsString, RoundsString],
        shape_number(Shape, Kind),
        string.to_int(TeamsString, Teams),
        Teams > 0,
        string.to_int(RoundsString, Rounds)
    then
        new_mailbox(Reports, !IO),
        run(Kind, Teams, warm_up, Reports, !IO),
        add_up(Reports, 0, _, 0, _, !IO),
        now(Empty0, !IO),
        run(Kind, Teams, 0, Reports, !IO),
        now(Empty1, !IO),
        add_up(Reports, 0, _, 0, _, !IO),
        now(T0, !IO),
        run(Kind, Teams, Rounds, Reports, !IO),
        now(T1, !IO),
        add_up(Reports, 0, Wrong, 0, Reported, !IO),
        report(Shape, Teams, Rounds, Reported, Wrong,
            (T1 - T0) - (Empty1 - Empty0), !IO)
    else
        io.write_string("usage: teambarrier central|tree TEAMS ROUNDS\n", !IO),
        io.set_exit_status(2, !IO)
    ).

:- pred shape_number(string::in, int::out) is semidet.

shape_number("central", 0).
shape_number("tree", 1).

    % Rounds barriers in each of Teams teams, each of which then sends the
    % number of messages it did not expect to Reports.
:- pred run(int::in, int::in, int::in, mailbox(int)::in, io::di, io::uo)
    is det.

run(Kind, Teams, Rounds, Reports, !IO) :-
    %$omp target teams num_teams(Teams)
    barriers(Kind, Rounds, Reports, !IO).

:- pred barriers(int::in, int::in, mailbox(int)::in, io::di, io::uo) is det.

barriers(Kind, Rounds, Reports, !IO) :-
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    team_mailbox(Me, Mine, !IO),
    ( if Kind = 0 then
        ( if Me = 0 then
            Up = [],
            mailboxes(1, All, Down, !IO),
            Waits = All - 1
        else
            team_mailbox(0, First, !IO),
            Up = [First],
            Down = [],
            Waits = 0
        )
    else
        ( if Me = 0 then
            Up = []
        else
            team_mailbox((Me - 1) / 2, Parent, !IO),
            Up = [Parent]
        ),
        children(Me, All, Down, !IO),
        Waits = list.length(Down)
    ),
    rounds(Mine, Up, Down, Waits, 0, Rounds, 0, Wrong, !IO),
    send(Reports, Wrong, !IO).

    % The mailboxes of teams From up to All.
:- pred mailboxes(int::in, int::in, list(mailbox(int))::out, io::di, io::uo)
    is det.

mailboxes(From, All, Boxes, !IO) :-
    ( if From < All then
        team_mailbox(From, Box, !IO),
        mailboxes(From + 1, All, Rest, !IO),
        Boxes = [Box | Rest]
    else
        Boxes = []
    ).

    % The mailboxes of a team's children in the tree.
:- pred children(int::in, int::in, list(mailbox(int))::out, io::di, io::uo)
    is det.

children(Me, All, Children, !IO) :-
    Left = 2 * Me + 1,
    Right = 2 * Me + 2,
    ( if Right < All then
        team_mailbox(Left, LeftBox, !IO),
        team_mailbox(Right, RightBox, !IO),
        Children = [LeftBox, RightBox]
    else if Left < All then
        team_mailbox(Left, LeftBox, !IO),
        Children = [LeftBox]
    else
        Children = []
    ).

    % Round R of Rounds: waits for Waits arrivals, arrives Up, waits for the
    % release from Up, and releases Down.
:- pred rounds(mailbox(int)::in, list(mailbox(int))::in,
    list(mailbox(int))::in, int::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.

rounds(Mine, Up, Down, Waits, R, Rounds, !Wrong, !IO) :-
    ( if R < Rounds then
        expect(Mine, Waits, 2 * R, !Wrong, !IO),
        send_all(Up, 2 * R, !IO),
        ( if Up = [] then
            true
        else
            expect(Mine, 1, 2 * R + 1, !Wrong, !IO)
        ),
        send_all(Down, 2 * R + 1, !IO),
        rounds(Mine, Up, Down, Waits, R + 1, Rounds, !Wrong, !IO)
    else
        true
    ).

    % Receives Left messages, counting those that are not Expected, or that
    % do not come.
:- pred expect(mailbox(int)::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.

expect(Mine, Left, Expected, !Wrong, !IO) :-
    ( if Left > 0 then
        receive(Mine, any, after(10000), Got, !IO),
        (
            Got = message(Number),
            ( if Number = Expected then
                true
            else
                !:Wrong = !.Wrong + 1
            )
        ;
            Got = timeout,
            !:Wrong = !.Wrong + 1
        ),
        expect(Mine, Left - 1, Expected, !Wrong, !IO)
    else
        true
    ).

:- pred send_all(list(mailbox(int))::in, int::in, io::di, io::uo) is det.

send_all([], _, !IO).
send_all([Box | Boxes], Message, !IO) :-
    send(Box, Message, !IO),
    send_all(Boxes, Message, !IO).

    % The reports of the last region: their number and the sum.
:- pred add_up(mailbox(int)::in, int::in, int::out, int::in, int::out,
    io::di, io::uo) is det.

add_up(Box, !Sum, !Count, !IO) :-
    receive(Box, any, after(0), Got, !IO),
    (
        Got = message(Number),
        add_up(Box, !.Sum + Number, !:Sum, !.Count + 1, !:Count, !IO)
    ;
        Got = timeout
    ).

%---------------------------------------------------------------------------%
% What C does for the program: the clock and printing.

:- pragma foreign_decl("C", local, "
#include <stdio.h>
#include <time.h>
").

:- pred now(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(Nanoseconds::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    Nanoseconds = (MR_Integer) ts.tv_sec * 1000000000 + ts.tv_nsec;
").

:- pred report(string::in, int::in, int::in, int::in, int::in, int::in,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    report(Shape::in, Teams::in, Rounds::in, Reported::in, Wrong::in,
        Nanoseconds::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"%-8s %5ld teams %8ld rounds, %5ld reported, %ld wrong, \"
        \"%10.2f us a barrier\\n\",
        Shape, (long) Teams, (long) Rounds, (long) Reported, (long) Wrong,
        Rounds > 0 ? Nanoseconds / 1000.0 / Rounds : 0.0);
    fflush(stdout);
").
