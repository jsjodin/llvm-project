% The unselected-message warning (design document, "The receive goal"): a
% predicate whose receive goals' arms for a mailbox, by the text of its
% term, leave out a constructor of its messages, which would stay queued,
% is warned about once, at the first of those arms' mailbox. An arm selects
% the constructor of its pattern's functor, whatever its arguments, or
% every message if its pattern is a variable of its own; a pattern's
% variable that occurs outside the goal is tested, so it selects no
% constructor. Of a type without a known set of constructors, only the
% latter selects every message. -Wno-unselected-message turns it off.
%
% RUN: merkc -fc1 -freceive-goal -fdebug-dump-determinism %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=warning: --implicit-check-not=error:
% RUN: merkc -fc1 -freceive-goal -Wno-unselected-message \
% RUN:   -fdebug-dump-determinism %s 2>&1 \
% RUN:   | FileCheck %s --check-prefix=OFF --implicit-check-not=warning:
% OFF: pred unselected.left_out/3

:- module unselected.
:- interface.
:- import_module bool, io, process.

:- type msg
    --->    start
    ;       data(int, string)
    ;       stop.

:- pred left_out(mailbox(msg)::in, io::di, io::uo) is det.
:- pred two_left_out(mailbox(msg)::in, mailbox(msg)::in, io::di, io::uo)
    is det.
:- pred numbers(mailbox(int)::in, io::di, io::uo) is det.
:- pred tested(mailbox(msg)::in, msg::in, io::di, io::uo) is det.
:- pred all_selected(mailbox(msg)::in, mailbox(int)::in, io::di, io::uo)
    is det.
:- pred another_receive(mailbox(msg)::in, io::di, io::uo) is det.
:- pred other_clause(bool::in, mailbox(msg)::in, io::di, io::uo) is det.

:- implementation.

left_out(Box, !IO) :-
    receive [io(!IO)] (
        select(Box, start) -> true
    ;
        select(Box, data(_, _)) -> true
    ).
% CHECK: unselected.m:[[@LINE-4]]:16: warning: no arm selects the messages of the mailbox Box whose functor is the constructor stop/0, so they stay queued for another receive [-Wunselected-message]

    % A pattern's arguments do not matter, only its functor. Only Inbox's
    % arms leave constructors out.
two_left_out(Inbox, Control, !IO) :-
    receive [io(!IO)] (
        select(Control, _) -> true
    ;
        select(Inbox, data(1, "one")) -> true
    ).
% CHECK: unselected.m:[[@LINE-2]]:16: warning: no arm selects the messages of the mailbox Inbox whose functor is one of the constructors start/0, stop/0, so they stay queued for another receive [-Wunselected-message]

numbers(Box, !IO) :-
    receive [io(!IO)] (
        select(Box, 1) -> true
    ;
        select(Box, 2) -> true
    ).
% CHECK: unselected.m:[[@LINE-4]]:16: warning: no arm selects every message of the mailbox Box, of type int, so the others stay queued for another receive [-Wunselected-message]

tested(Box, Expected, !IO) :-
    receive [io(!IO)] (
        select(Box, Expected) -> true
    ;
        select(Box, start) -> true
    ).
% CHECK: unselected.m:[[@LINE-4]]:16: warning: no arm selects the messages of the mailbox Box whose functor is one of the constructors data/2, stop/0, so they stay queued for another receive [-Wunselected-message]

all_selected(Box, Numbers, !IO) :-
    receive [io(!IO)] (
        select(Box, start) -> true
    ;
        select(Numbers, N) -> io.write_int(N, !IO)
    ;
        select(Box, data(_, _)) -> true
    ;
        select(Box, stop) -> true
    ).

    % Another receive goal of the predicate selects what the first leaves.
another_receive(Box, !IO) :-
    receive [io(!IO)] (
        select(Box, start) -> true
    ),
    receive [io(!IO)] (
        select(Box, data(_, _)) -> true
    ;
        select(Box, stop) -> true
    ).

    % Or one in another clause of it, for a mailbox written the same way.
other_clause(no, Box, !IO) :-
    receive [io(!IO)] (
        select(Box, stop) -> true
    ).
other_clause(yes, Box, !IO) :-
    receive [io(!IO)] (
        select(Box, M) -> io.write(M, !IO)
    ).
