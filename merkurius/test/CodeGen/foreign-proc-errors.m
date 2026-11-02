% clang's errors in a foreign_proc's C code are reported at their lines in
% the Mercury source, through #line directives.
%
% RUN: not merkc -fc1 -emit-obj %s -o %t.o 2>&1 | FileCheck %s

:- module foreign_proc_errors.
:- interface.
:- pred p(int::in, int::out) is det.
:- implementation.
:- pragma foreign_proc("C",
    p(X::in, Y::out),
    [will_not_call_mercury, promise_pure],
"
    Y = X +
        undeclared;
").
% CHECK: foreign-proc-errors.m:[[@LINE-2]]:9: error: use of undeclared identifier 'undeclared'
% CHECK: merkc: error: clang could not compile the module's foreign code
