% Misuses of team mailboxes that end the program, chosen by the first
% argument (omp-target-team-mailbox-misuse-gpu.test).
:- module omp_target_team_mailbox_misuse.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process.

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if Arguments = ["range"] then
        %$omp target teams num_teams(2)
        out_of_range(!IO)
    else if Arguments = ["other"] then
        %$omp target teams num_teams(2)
        receive_other(!IO)
    else
        io.write_string("range or other\n", !IO)
    ).

    % Team 1 asks for the mailbox of team 5 of 2.
:- pred out_of_range(io::di, io::uo) is det.

out_of_range(!IO) :-
    team_number(Me, !IO),
    ( if Me = 1 then
        team_mailbox(5, Box : mailbox(int), !IO),
        send(Box, 1, !IO)
    else
        true
    ).

    % Team 1 receives from team 0's mailbox.
:- pred receive_other(io::di, io::uo) is det.

receive_other(!IO) :-
    team_number(Me, !IO),
    ( if Me = 1 then
        team_mailbox(0, Box : mailbox(int), !IO),
        receive(Box, any, after(0), _ : received(int), !IO)
    else
        true
    ).
