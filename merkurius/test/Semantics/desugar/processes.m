% Process declarations, with -fprocess-declarations (design document,
% "Processes and mailboxes"): each declares the kind's predicate, with its
% arguments, a mailbox for its messages and the io state, and spawn_Kind,
% whose clause spawns a process that calls the kind's predicate, qualified
% by the module. Both are declared where the declaration is, here in the
% interface, and spawn_Kind's clause is the module's own.
% CodeGen/process-declarations.m runs them.
%
% RUN: merkc -fc1 -fprocess-declarations -fdebug-dump-semantic-form %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=warning: --implicit-check-not=error:
% RUN: merkc -fc1 -fprocess-declarations -fdebug-dump-resolved-names %s 2>&1 \
% RUN:   | FileCheck %s --check-prefix=SYMBOLS --implicit-check-not=error:

:- module processes.
:- interface.
:- import_module io, list, process.

:- type msg ---> m.

:- process counter(int, string) receives msg.
% CHECK:      Predicate spawn_counter(A1, A2, Mailbox, IO0, IO)
% CHECK-NEXT: | Conj #3
% CHECK-NEXT: | | Unify #1 $5 = pred(Self::in, Self0::di, Self1::uo) is det
% CHECK-NEXT: | | | Call #0 processes.counter(A1, A2, Self, Self0, Self1)
% CHECK-NEXT: | | Call #2 process.spawn($5, Mailbox, IO0, IO)
% SYMBOLS:      [[@LINE-6]]:4 pred processes.counter/5 exported
% SYMBOLS-NEXT: [[@LINE-7]]:4 pred processes.spawn_counter/5 exported

:- process logger receives list(T).
% CHECK:      Predicate spawn_logger(Mailbox, IO0, IO)
% CHECK-NEXT: | Conj #3
% CHECK-NEXT: | | Unify #1 $3 = pred(Self::in, Self0::di, Self1::uo) is det
% CHECK-NEXT: | | | Call #0 processes.logger(Self, Self0, Self1)
% CHECK-NEXT: | | Call #2 process.spawn($3, Mailbox, IO0, IO)
% SYMBOLS:      [[@LINE-6]]:4 pred processes.logger/3 exported
% SYMBOLS-NEXT: [[@LINE-7]]:4 pred processes.spawn_logger/3 exported

:- implementation.

counter(_, _, _, !IO).
logger(_, !IO).
