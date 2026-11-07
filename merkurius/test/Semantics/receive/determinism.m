% A receive goal is det if each of its arms' goals is; a selective receive
% waits for a message that an arm selects, rather than failing, so an arm
% that can fail is what makes it semidet (design document, "The receive
% goal").
%
% RUN: not merkc -fc1 -freceive-goal -fdebug-dump-determinism %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module determinism.
:- interface.
:- import_module io, process.
:- pred selective(mailbox(int)::in, io::di, io::uo) is det.
:- pred can_fail(mailbox(int)::in, io::di, io::uo) is det.
:- implementation.
:- import_module int.

selective(Box, !IO) :-
    receive [io(!IO)] ( select(Box, 3) -> true ).

can_fail(Box, !IO) :-
    receive [io(!IO)] ( select(Box, N) -> N > 0 ).
% CHECK: determinism.m:13:9: error: the determinism declaration of can_fail(in, di, uo) is not satisfied: it is declared det, but inferred semidet
% CHECK: determinism.m:[[@LINE-2]]:43: because: in the case receive_message/1 of the switch on Received: the call to >(in, in) can fail
