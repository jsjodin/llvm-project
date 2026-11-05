% !, !. and !: of anything but a variable are constructors in a data-term, so a
% place that wants a state variable or a variable rejects them. mmc rejects each
% of these too.
%
% RUN: not merkc -fc1 -fdebug-dump-semantic-form %s 2>&1 | FileCheck %s

:- module state_var_of_non_variable.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> f ; (! t).
:- type r ---> r(x :: int).

:- pred quantified_some(io::di, io::uo) is det.
quantified_some(!IO) :-
    some [!f] true.
% CHECK: [[@LINE-1]]:11: error: {{.*}}a variable or a state variable

:- pred quantified_all(io::di, io::uo) is det.
quantified_all(!IO) :-
    all [!.f] true.
% CHECK: [[@LINE-1]]:10: error: {{.*}}a variable or a state variable

:- pred traced(io::di, io::uo) is det.
traced(!IO) :-
    trace [io(!f)] true.
% CHECK: [[@LINE-1]]:15: error: {{.*}}a state variable !X

:- pred traced_state(io::di, io::uo) is det.
traced_state(!IO) :-
    trace [state(f, !f)] true.
% CHECK: [[@LINE-1]]:21: error: {{.*}}a state variable !X

:- pred tried(io::di, io::uo) is det.
tried(!IO) :-
    try [io(!f)] true
    then true.
% CHECK: [[@LINE-2]]:13: error: {{.*}}a state variable !X

:- pred updated(r::in, r::out) is det.
updated(!R) :-
    !f ^ x := 1.
% CHECK: [[@LINE-1]]:5: error: {{.*}}DCG goal

main(!IO).
