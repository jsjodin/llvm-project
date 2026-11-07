% Trace goals (reference manual, "Trace goals"). One whose compile_time
% condition does not hold is left out: flag(Name) holds with
% --trace-flag=Name, tracelevel never, and of the grade tests c and mlds,
% as in mmc's default grade. One with a run_time condition is in merk.ite,
% whose condition asks the runtime whether each environment variable
% existed when the program started, whatever its value, even empty. Its io
% state has no value to pass. One whose goal has more than one solution
% is in merk.commit. The output is mmc's for the same program, with and
% without --trace-flag=off.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && env -u TRACE_GOALS_UNSET TRACE_GOALS_SET= %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && env -u TRACE_GOALS_UNSET TRACE_GOALS_SET= %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 --trace-flag=off %s -o %t3 && env -u TRACE_GOALS_UNSET TRACE_GOALS_SET= %t3 | FileCheck %s --check-prefix=FLAG --match-full-lines --strict-whitespace

:- module trace_goals.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- pred two(int::out) is multi.
two(10).
two(20).

    % In a function, which has no io state of its own.
:- func double(int) = int.
double(X) = Y :-
    trace [io(!IO)] (
        io.write_string("double ", !IO),
        io.write_int(X, !IO),
        io.nl(!IO)
    ),
    Y = X * 2.
% MERK-LABEL: merk.proc private @trace_goals.double_f1_0(%arg0: i64) -> i64
% MERK: merk.call @io.write_string_3_0(
% MERK: merk.call @io.write_int_3_0(%arg0) : (i64) -> ()
% MERK: merk.call @io.nl_2_0() : () -> ()
% MERK: arith.muli

    % Its goal's first solution.
:- pred first(int::in) is det.
first(X) :-
    trace [io(!IO)] (
        two(Z),
        io.write_string("first ", !IO),
        io.write_int(X + Z, !IO),
        io.nl(!IO)
    ).
% MERK-LABEL: merk.proc private @trace_goals.first_1_0(
% MERK: merk.commit {
% MERK: merk.nondet_call @trace_goals.two_1_0()

    % In a semidet procedure, after a test.
:- pred positive(int::in) is semidet.
positive(X) :-
    X > 0,
    trace [io(!IO)] (
        io.write_string("positive ", !IO),
        io.write_int(X, !IO),
        io.nl(!IO)
    ).

    % In a nondet procedure, for each solution of the call before it.
:- pred member_traced(int::out, list(int)::in) is nondet.
member_traced(X, [H | T]) :-
    trace [io(!IO)] (
        io.write_string("visit ", !IO),
        io.write_int(H, !IO),
        io.nl(!IO)
    ),
    ( X = H ; member_traced(X, T) ).

:- pred conditions(io::di, io::uo) is det.
conditions(!IO) :-
    trace [compile_time(flag("off")), io(!T)] (
        io.write_string("flag off\n", !T) ),
    trace [compile_time(not flag("off")), io(!T)] (
        io.write_string("not flag off\n", !T) ),
    trace [compile_time(grade(c) and grade(mlds)), io(!T)] (
        io.write_string("c and mlds\n", !T) ),
    trace [compile_time(grade(llds) or grade(java)), io(!T)] (
        io.write_string("llds or java\n", !T) ),
    trace [compile_time(tracelevel(shallow)), io(!T)] (
        io.write_string("shallow\n", !T) ),
    trace [run_time(env("TRACE_GOALS_UNSET")), io(!T)] (
        io.write_string("unset\n", !T) ),
    trace [run_time(not env("TRACE_GOALS_UNSET")), io(!T)] (
        io.write_string("not unset\n", !T) ),
    trace [run_time(env("TRACE_GOALS_SET") or env("TRACE_GOALS_UNSET")),
            io(!T)] (
        io.write_string("set or unset\n", !T) ),
    trace [run_time(env("TRACE_GOALS_SET") and env("TRACE_GOALS_UNSET")),
            io(!T)] (
        io.write_string("set and unset\n", !T) ),
    trace [compile_time(grade(c)), run_time(env("TRACE_GOALS_SET")),
            io(!T)] (
        io.write_string("both\n", !T) ),
    io.write_string("conditions done\n", !IO).
% MERK-LABEL: merk.proc private @trace_goals.conditions_2_0()
% MERK-NOT: "flag off\0A"
% MERK: merk.constant "not flag off\0A"
% MERK: merk.constant "c and mlds\0A"
% MERK-NEXT: merk.call @io.write_string_3_0(
% MERK-NEXT: merk.ite {
% MERK-NEXT: %[[NAME:.*]] = merk.constant "TRACE_GOALS_UNSET" : !merk.string
% MERK-NEXT: %[[SET:.*]] = merk.foreign_call @merk_env_at_start(%[[NAME]]) can_fail : (!merk.string) -> i1
% MERK-NEXT: merk.guard %[[SET]]
% MERK-NEXT: merk.yield
% MERK-NEXT: } then {
% MERK-NEXT: merk.constant "unset\0A"
% MERK-NEXT: merk.call @io.write_string_3_0(
% MERK-NEXT: merk.yield
% MERK-NEXT: } else {
% MERK-NEXT: merk.yield
% MERK-NEXT: }
% MERK: arith.xori
% MERK: merk.constant "TRACE_GOALS_SET"
% MERK: merk.constant "TRACE_GOALS_UNSET"
% MERK: arith.ori
% MERK: merk.constant "TRACE_GOALS_SET"
% MERK: merk.constant "TRACE_GOALS_UNSET"
% MERK: arith.andi
% MERK-NOT: "llds or java\0A"
% MERK-NOT: "shallow\0A"
% MERK: merk.constant "conditions done\0A"

main(!IO) :-
    io.write_int(double(21), !IO),
    io.nl(!IO),
    first(1),
    ( if positive(3) then io.write_string("yes\n", !IO) else true ),
    ( if positive(-3) then
        io.write_string("yes\n", !IO)
    else
        io.write_string("no\n", !IO)
    ),
    solutions(
        ( pred(X::out) is nondet :-
            member_traced(X, [1, 2, 3]),
            trace [io(!T)] (
                io.write_string("lambda ", !T),
                io.write_int(X, !T),
                io.nl(!T)
            )
        ), L),
    io.write(L, !IO),
    io.nl(!IO),
    conditions(!IO).

% OUTPUT:double 21
% OUTPUT-NEXT:42
% OUTPUT-NEXT:first 11
% OUTPUT-NEXT:positive 3
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:no
% OUTPUT-NEXT:visit 1
% OUTPUT-NEXT:lambda 1
% OUTPUT-NEXT:visit 2
% OUTPUT-NEXT:lambda 2
% OUTPUT-NEXT:visit 3
% OUTPUT-NEXT:lambda 3
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:not flag off
% OUTPUT-NEXT:c and mlds
% OUTPUT-NEXT:not unset
% OUTPUT-NEXT:set or unset
% OUTPUT-NEXT:both
% OUTPUT-NEXT:conditions done
% OUTPUT-EMPTY:

% FLAG:double 21
% FLAG:[1, 2, 3]
% FLAG-NEXT:flag off
% FLAG-NEXT:c and mlds
% FLAG-NEXT:not unset
% FLAG-NEXT:set or unset
% FLAG-NEXT:both
% FLAG-NEXT:conditions done
% FLAG-EMPTY:
