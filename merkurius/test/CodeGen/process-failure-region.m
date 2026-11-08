% A failure in a parallel region ends the program, not only its process, as
% libomp's team cannot be abandoned (design document, "Exceptions, links and
% monitors"). Without -fopenmp there is no region, and process-failure.m
% tests what happens then, so this has no sequential run.
%
% REQUIRES: libomp
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t
% RUN: %libomp-env OMP_NUM_THREADS=4 not %t 2> %t.err | FileCheck %s --check-prefix=NOTHING --allow-empty
% RUN: FileCheck %s --check-prefix=REPORT --match-full-lines --strict-whitespace < %t.err

:- module process_failure_region.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

:- func sum(int, int) = int.
sum(N, D) = S :-
    some [!S] (
        !:S = 0,
        %$omp parallel do reduction((+) : S)
        ( for(I, 1, N) do !:S = !.S + I // D ),
        S = !.S
    ).

:- pred divider(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
divider(Reply, Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = yes(D),
        send(Reply, sum(1000, D), !IO)
    ;
        Got = no
    ).

main(!IO) :-
    new_mailbox(Reply, !IO),
    spawn(divider(Reply), Divider, !IO),
    send(Divider, 0, !IO),
    receive(Reply, any, after(20000), Got, !IO),
    io.write_line(Got, !IO).

% NOTHING-NOT:{{.}}

% REPORT:Uncaught Mercury exception:
% REPORT-NEXT:domain_error("int.\'//\': division by zero")
