% Process declarations, with -fprocess-declarations (design document,
% "Processes and mailboxes"): each declares the kind's predicate, with its
% arguments, a mailbox for its messages and the io state, and spawn_Kind,
% whose clause spawns a process that calls the kind's predicate, qualified
% by the module, and spawn_Kind_at, which spawns at a place: at the host,
% through spawn_Kind, and at a GPU, through the kind's kernel if the kind
% runs on a GPU, with its arguments' words, and otherwise with a software
% error (docs/INPROGRESS.md, GPU processes without omp target, step 4c).
% They are declared where the declaration is, here in the interface, and
% their clauses are the module's own.
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
% CHECK:      Predicate spawn_counter_at(Place, A1, A2, Mailbox, IO0, IO)
% CHECK-NEXT: | Disj #8
% CHECK-NEXT: | | Conj #6
% CHECK-NEXT: | | | Unify #0 Place = process.host
% CHECK-NEXT: | | | Call #1 processes.spawn_counter(A1, A2, Mailbox, IO0, IO)
% CHECK-NEXT: | | Conj #7
% CHECK-NEXT: | | | Unify #2 Place = process.gpu(GPU)
% CHECK-NEXT: | | | Unify #3 $7 = "processes.spawn_counter_at"
% CHECK-NEXT: | | | Unify #4 $8 = "counter"
% CHECK-NEXT: | | | Call #5 process.spawn_host_kind_at_gpu(GPU, $7, $8, Mailbox, IO0, IO)
% SYMBOLS:      [[@LINE-16]]:4 pred processes.counter/5 exported
% SYMBOLS-NEXT: [[@LINE-17]]:4 pred processes.spawn_counter/5 exported
% SYMBOLS-NEXT: [[@LINE-18]]:4 pred processes.spawn_counter_at/6 exported

:- process logger receives list(T).
% CHECK:      Predicate spawn_logger(Mailbox, IO0, IO)
% CHECK-NEXT: | Conj #3
% CHECK-NEXT: | | Unify #1 $3 = pred(Self::in, Self0::di, Self1::uo) is det
% CHECK-NEXT: | | | Call #0 processes.logger(Self, Self0, Self1)
% CHECK-NEXT: | | Call #2 process.spawn($3, Mailbox, IO0, IO)
% SYMBOLS:      [[@LINE-6]]:4 pred processes.logger/3 exported
% SYMBOLS-NEXT: [[@LINE-7]]:4 pred processes.spawn_logger/3 exported

:- process summer(int, float) receives msg on gpu.
% CHECK:      Predicate spawn_summer_at(Place, A1, A2, Mailbox, IO0, IO)
% CHECK-NEXT: | Disj #12
% CHECK-NEXT: | | Conj #10
% CHECK-NEXT: | | | Unify #0 Place = process.host
% CHECK-NEXT: | | | Call #1 processes.spawn_summer(A1, A2, Mailbox, IO0, IO)
% CHECK-NEXT: | | Conj #11
% CHECK-NEXT: | | | Unify #2 Place = process.gpu(GPU)
% CHECK-NEXT: | | | Unify #3 $7 = "processes.spawn_summer_at"
% CHECK-NEXT: | | | Unify #4 $8 = "processes"
% CHECK-NEXT: | | | Unify #5 $9 = "summer"
% CHECK-NEXT: | | | Unify #8 $10 = process.gpu_words(A2, $11)
% CHECK-NEXT: | | | Unify #7 $11 = process.gpu_words(A1, $12)
% CHECK-NEXT: | | | Unify #6 $12 = process.no_gpu_words
% CHECK-NEXT: | | | Call #9 process.spawn_kernel(GPU, $7, $8, $9, $10, Mailbox, IO0, IO)

:- implementation.

summer(_, _, _, !IO).
counter(_, _, _, !IO).
logger(_, !IO).
