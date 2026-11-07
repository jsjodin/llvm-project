% The determinism of trace goals (reference manual, "Trace goals"), as mmc
% checks it: a trace goal must be det or cc_multi. Only its first solution
% is wanted, so one whose goal has more than one is committed choice, and
% is det to the goals around it; one that can fail is an error, also when
% its compile_time condition does not hold, and makes the procedure's
% determinism one that can fail. Its committed choice determinism is the
% one reported.
%
% RUN: not merkc -fc1 -fdebug-dump-determinism %s 2>%t.err | FileCheck %s
% RUN: FileCheck --check-prefix=ERR --implicit-check-not=error: --implicit-check-not=because: %s < %t.err
% RUN: %if mmc %{ %compare-switches --determinism %s %}

:- module trace_goals.
:- interface.
:- import_module io.

:- pred two(int::out) is multi.
:- pred multi(int::in) is det.
:- pred semi(int::in) is det.
% ERR: trace-goals.m:[[@LINE-1]]:9: error: the determinism declaration of semi(in) is not satisfied: it is declared det, but inferred semidet
% ERR: trace-goals.m:{{[0-9]+}}:23: because: the call to >(in, in) can fail
:- pred nondet(int::in) is det.
% ERR: trace-goals.m:[[@LINE-1]]:9: error: the determinism declaration of nondet(in) is not satisfied: it is declared det, but inferred semidet
% ERR: trace-goals.m:{{[0-9]+}}:23: because: the call to member(out, in) can fail
:- pred deleted(int::in) is det.
% ERR: trace-goals.m:[[@LINE-1]]:9: error: the determinism declaration of deleted(in) is not satisfied: it is declared det, but inferred semidet
:- pred waits(int::out) is det.
% ERR: trace-goals.m:[[@LINE-1]]:9: error: the determinism declaration of waits(out) is not satisfied: it is declared det, but inferred failure
:- pred erroneous(int::in) is det.
:- pred in_semidet(int::in) is semidet.

:- implementation.
:- import_module int, list, require.

two(1).
two(2).

multi(X) :-
    trace [io(!IO)] ( two(Z), io.write_int(X + Z, !IO) ).
% CHECK-LABEL: pred trace_goals.multi/1 mode 0: multi(in) is det, inferred det
% CHECK-NEXT: trace [det]
% CHECK-NEXT: call pred trace_goals.two/1 mode 0: two(Z) [multi]

semi(X) :-
    trace [io(!IO)] ( X > 2, io.write_int(X, !IO) ).
% CHECK-LABEL: pred trace_goals.semi/1 mode 0: semi(in) is det, inferred semidet
% ERR: trace-goals.m:[[@LINE-2]]:5: error: the trace goal has determinism semidet, but must be det or cc_multi

nondet(X) :-
    trace [io(!IO)] ( list.member(Z, [X, X + 1]), io.write_int(Z, !IO) ).
% CHECK-LABEL: pred trace_goals.nondet/1 mode 0: nondet(in) is det, inferred semidet
% ERR: trace-goals.m:[[@LINE-2]]:5: error: the trace goal has determinism cc_nondet, but must be det or cc_multi

deleted(X) :-
    trace [compile_time(flag("never"))] ( X > 100 -> fail ; true ).
% CHECK-LABEL: pred trace_goals.deleted/1 mode 0: deleted(in) is det, inferred semidet
% ERR: trace-goals.m:[[@LINE-2]]:5: error: the trace goal has determinism semidet, but must be det or cc_multi

    % Y = 4 runs first, as the trace goal may not bind Y, so Y = 3 is a
    % test that cannot succeed.
waits(Y) :-
    trace [io(!IO)] ( Y = 3, io.write_int(Y, !IO) ),
    Y = 4.
% CHECK-LABEL: pred trace_goals.waits/1 mode 0: waits(out) is det, inferred failure
% ERR: trace-goals.m:[[@LINE-3]]:5: error: the trace goal has determinism failure, but must be det or cc_multi
% ERR: trace-goals.m:[[@LINE-4]]:27: warning: the unification of Y and 3 cannot succeed: Y has inst unique(4)

erroneous(X) :-
    trace [io(!IO)] (
        io.write_int(X, !IO),
        ( X > 100 -> true ; error("small") )
    ).
% CHECK-LABEL: pred trace_goals.erroneous/1 mode 0: erroneous(in) is det, inferred det

in_semidet(X) :-
    X > 0,
    trace [io(!IO)] ( two(Z), io.write_int(Z, !IO) ).
% CHECK-LABEL: pred trace_goals.in_semidet/1 mode 0: in_semidet(in) is semidet, inferred semidet
