% A new state variable that hides one in scope warns, as in mmc, though
% the reference manual says nothing about it: that of some [!X] and that of
% a trace goal's io(!X), but not a lambda expression's head, whose !X is
% its own. One of another name does not warn.
%
% RUN: merkc -fc1 -fdebug-dump-semantic-form %s 2>&1 >/dev/null | FileCheck %s --implicit-check-not=warning: --implicit-check-not=error:

:- module state_var_shadowing.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(!IO) :-
    some [!IO] (
% CHECK: state-var-shadowing.m:[[@LINE-1]]:11: warning: the new state variable !IO shadows the one outside
        !:IO = 1,
        _ = !.IO
    ),
    some [!S] (
        !:S = 1,
        some [!S] ( !:S = 2, _ = !.S ),
% CHECK: state-var-shadowing.m:[[@LINE-1]]:15: warning: the new state variable !S shadows the one outside
        _ = !.S
    ),
    P = (pred(!.IO::di, !:IO::uo) is det :- io.write_int(1, !IO)),
    P(!IO),
    trace [io(!IO)] ( io.write_int(2, !IO) ),
% CHECK: state-var-shadowing.m:[[@LINE-1]]:15: warning: the new state variable !IO shadows the one outside
    trace [io(!T)] ( io.write_int(3, !T) ),
    io.nl(!IO).
