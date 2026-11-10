% A target teams region with no loop runs its goal in every team
% (docs/INPROGRESS.md, Processes as teams): a generic kernel, with omp.teams
% in omp.target, as Flang lowers target teams. The number of teams is a
% host_eval value on the host, checked before the region starts, and read
% from its map on the device. Each team sends its number to a mailbox on the
% host. Without OpenMP the region runs once, as team 0 of 1.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null
% RUN: merkc -O0 -fopenmp %libomp-flags %s -o %t0 && %libomp-env %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t2 && %libomp-env %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=SEQUENTIAL --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_teams.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, int, list, process.

main(!IO) :-
    new_mailbox(Box, !IO),
    N = 4,
    %$omp target teams num_teams(N)
    report(Box, !IO),
    collect(Box, [], Numbers, !IO),
    list.sort(Numbers, Sorted),
    io.write_string("asked for ", !IO),
    io.write_int(N, !IO),
    io.write_string(", got ", !IO),
    io.print_line(Sorted, !IO),
    new_mailbox(Some, !IO),
    %$omp target teams
    some_teams(Some, !IO),
    collect_bools(Some, [], Answers, !IO),
    io.write_string("teams by default, at least 1: ", !IO),
    io.print_line(Answers, !IO),
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    io.write_string("outside: team ", !IO),
    io.write_int(Me, !IO),
    io.write_string(" of ", !IO),
    io.write_int(All, !IO),
    io.nl(!IO).

:- pred report(mailbox(int)::in, io::di, io::uo) is det.

report(Box, !IO) :-
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    ( if Me = 0 then
        io.write_string("teams: ", !IO),
        io.write_int(All, !IO),
        io.nl(!IO)
    else
        true
    ),
    send(Box, Me, !IO).

:- pred some_teams(mailbox(bool)::in, io::di, io::uo) is det.

some_teams(Box, !IO) :-
    team_number(Me, !IO),
    number_of_teams(All, !IO),
    ( if Me = 0 then
        send(Box, ( if All >= 1 then yes else no ), !IO)
    else
        true
    ).

:- pred collect(mailbox(int)::in, list(int)::in, list(int)::out,
    io::di, io::uo) is det.

collect(Box, !Numbers, !IO) :-
    receive(Box, any, after(0), Got, !IO),
    (
        Got = message(Number),
        collect(Box, [Number | !.Numbers], !:Numbers, !IO)
    ;
        Got = timeout
    ).

:- pred collect_bools(mailbox(bool)::in, list(bool)::in, list(bool)::out,
    io::di, io::uo) is det.

collect_bools(Box, !Answers, !IO) :-
    receive(Box, any, after(0), Got, !IO),
    (
        Got = message(Answer),
        collect_bools(Box, [Answer | !.Answers], !:Answers, !IO)
    ;
        Got = timeout
    ).

% MERK-LABEL: merk.proc @omp_target_teams.main_2_0()
% MERK: merk.foreign_call @merk_check_teams(%[[N:.*]]) : (i64) -> ()
% MERK: %[[COUNT:.*]] = arith.trunci %[[N]] : i64 to i32
% MERK: %[[MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i32) map_clauses(to) capture(ByCopy)
% MERK: omp.target kernel_type(generic) host_eval(%[[COUNT]] -> %[[EVAL:.*]] : i32) map_entries(%{{.*}} -> %{{.*}}, %[[MAP]] -> %{{.*}} : !llvm.ptr, !llvm.ptr) {
% MERK: omp.teams num_teams( to %[[EVAL]] : i32) {
% MERK-NEXT: merk.call @omp_target_teams.report_3_0
% MERK: omp.terminator
% MERK: omp.terminator
% MERK-NEXT: } {omp.combined}
% MERK-NOT: merk_check_teams
% MERK: omp.target kernel_type(generic) map_entries(%{{.*}} -> %{{.*}} : !llvm.ptr) {
% MERK-NEXT: %{{.*}} = builtin.unrealized_conversion_cast
% MERK-NEXT: omp.teams {
% MERK-NEXT: merk.call @omp_target_teams.some_teams_3_0
% MERK: merk.foreign_call @omp_get_team_num() : () -> i32
% MERK: merk.foreign_call @omp_get_num_teams() : () -> i32
% MERK-LABEL: merk.proc private @omp_target_teams.report_3_0(
% MERK: merk.foreign_call @omp_get_team_num() : () -> i32
% MERK: merk.foreign_call @omp_get_num_teams() : () -> i32

% DEVICE-LABEL: merk.proc @omp_target_teams.main_2_0()
% DEVICE: omp.target kernel_type(generic) map_entries(%{{.*}} -> %{{.*}}, %{{.*}} -> %[[ARG:[^ ]*]] : !llvm.ptr, !llvm.ptr) {
% DEVICE: %[[COUNT:.*]] = llvm.load %[[ARG]] : !llvm.ptr -> i32
% DEVICE: omp.teams num_teams( to %[[COUNT]] : i32) {

% OUTPUT:teams: 4
% OUTPUT-NEXT:asked for 4, got [0, 1, 2, 3]
% OUTPUT-NEXT:teams by default, at least 1: [yes]
% OUTPUT-NEXT:outside: team 0 of 1
% OUTPUT-NOT:{{.}}

% SEQUENTIAL:teams: 1
% SEQUENTIAL-NEXT:asked for 4, got [0]
% SEQUENTIAL-NEXT:teams by default, at least 1: [yes]
% SEQUENTIAL-NEXT:outside: team 0 of 1
% SEQUENTIAL-NOT:{{.}}
