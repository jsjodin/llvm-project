% For omp-target-teams-too-many-gpu.test: 2000 teams, more than run at once
% on a GPU, first without team mailboxes, which runs, and then with them,
% which is an error, as some teams would wait for teams that are not running.

:- module omp_target_teams_too_many.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

main(!IO) :-
    new_mailbox(Counts, !IO),
    %$omp target teams num_teams(2000)
    send(Counts, 1, !IO),
    add_up(Counts, 0, Total, !IO),
    io.write_string("teams without team mailboxes: ", !IO),
    io.write_int(Total, !IO),
    io.nl(!IO),
    io.flush_output(!IO),
    %$omp target teams num_teams(2000)
    to_first(!IO),
    io.write_string("not reached\n", !IO).

:- pred to_first(io::di, io::uo) is det.

to_first(!IO) :-
    team_mailbox(0, First, !IO),
    send(First, 1, !IO).

:- pred add_up(mailbox(int)::in, int::in, int::out, io::di, io::uo) is det.

add_up(Box, !Total, !IO) :-
    receive(Box, any, after(0), Got, !IO),
    (
        Got = message(Number),
        add_up(Box, !.Total + Number, !:Total, !IO)
    ;
        Got = timeout
    ).
