% A mode declaration whose `with_inst` inst is a function's is a function's
% mode, though it has no result: the inst gives the result's mode
% (reference manual, "Predicate and function mode declarations").
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module with_inst_function.
:- interface.

:- type adder == (func(int, int) = int).
:- inst adder == (func(in, in) = out is det).
:- type stepper == pred(int, int).
:- inst stepper == (pred(in, out) is semidet).

:- func add3(int) `with_type` adder.
:- mode add3(in) `with_inst` adder.
:- pred step(int) `with_type` stepper.
:- mode step(in) `with_inst` stepper.

:- implementation.
:- import_module int.

add3(A, B, C) = A + B + C.
step(A, B, C) :-
    A > 0,
    C = A + B.

% CHECK: 17:9 func with_inst_function.add3/3 mode 0: add3(in, in, in) = out is det
% CHECK: 19:9 pred with_inst_function.step/3 mode 0: step(in, in, out) is semidet
