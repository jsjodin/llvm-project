% Errors in process declarations, read with -fprocess-declarations (design
% document, "Processes and mailboxes"). An item with an error is left out.
%
% RUN: not merkc -fc1 -fprocess-declarations -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module process_errors.
:- interface.

:- process a(int) receives msg on cpu.
% CHECK: process-errors.m:[[@LINE-1]]:35: error: expected gpu, the place where the process kind runs, but found 'cpu'

:- process b(int).
% CHECK: process-errors.m:[[@LINE-1]]:12: error: expected a process kind, followed by receives and its message types, but found 'b(int)'

:- process c(int) on gpu.
% CHECK: process-errors.m:[[@LINE-1]]:12: error: expected a process kind, followed by receives and its message types, but found 'c(int)'

:- process m.d(int) receives msg.
% CHECK: process-errors.m:[[@LINE-1]]:12: error: expected a process kind, a name with its argument types, not module-qualified, but found 'm.d(int)'

:- process 3 receives msg.
% CHECK: process-errors.m:[[@LINE-1]]:12: error: expected a process kind, a name with its argument types, not module-qualified, but found '3'

:- process E receives msg.
% CHECK: process-errors.m:[[@LINE-1]]:12: error: expected a process kind, a name with its argument types, not module-qualified, but found 'E'

:- process f("x") receives msg.
% CHECK: process-errors.m:[[@LINE-1]]:14: error: expected a type, but found '"x"'

:- process g(int) receives msg, 42.
% CHECK: process-errors.m:[[@LINE-1]]:33: error: expected a type, but found '42'

:- process(h).
% CHECK: process-errors.m:[[@LINE-1]]:12: error: expected a process kind, followed by receives and its message types, but found 'h'

    % A declaration that does not begin the item is not one, and process,
    % receives and on are not operators there.
:- interface, process i(int) receives msg.
% CHECK: process-errors.m:[[@LINE-1]]:23: error: expected an operator or '.', but found 'i'
