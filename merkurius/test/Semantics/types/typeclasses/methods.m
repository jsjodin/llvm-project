% Instance methods are auxiliary predicates and functions, each checked by
% its method's signature with the instance's types in place of the type
% class's parameters (reference manual, "Instance declarations"). The
% instance's constraints are assumed, and so are the method's own. A
% method defined by name, pred(m/N) is p, is the clause m(X1, ..., XN) :-
% p(X1, ..., XN), and func(m/N) is f is m(X1, ..., XN) = f(X1, ..., XN),
% as mmc checks them: the predicate or function may be more general.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --check-prefix=USE %s
% RUN: %if mmc %{ %compare-types %s %}

:- module methods.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, string.

:- typeclass show(T) where [
    func show(T) = string,
    pred show_to(T::in, string::out) is det
].
:- typeclass loud(T) <= show(T) where [ func loud(T) = string ].
:- typeclass conv(A, B) where [
    func conv(A) = B,
    func with_shown(A, B, C) = string <= show(C)
].

main(0).

:- instance show(int) where [
    show(X) = string.int_to_string(X),
% CHECK: clause at [[@LINE-1]]:5 of func methods.show/1 in instance methods.show(int): func methods.show(int) = string
    pred(show_to/2) is any_to_empty
% CHECK: clause at [[@LINE-1]]:5 of pred methods.show_to/2 in instance methods.show(int): pred methods.show_to(int, string)
% CHECK-NEXT: [[@LINE-2]]:24 any_to_empty -> pred methods.any_to_empty/2
].
:- pred any_to_empty(T::in, string::out) is det.
any_to_empty(_, "").

% The instance's constraint gives show(T), and the recursion is by the
% instance itself.
:- instance show(list(T)) <= show(T) where [
    show([]) = "",
    show([X | Xs]) = show(X) ++ show(Xs),
% CHECK: clause at [[@LINE-1]]:5 of func methods.show/1 in instance methods.show(list.list(T)): func methods.show(list.list(T)) = string <= (methods.show(T))
% CHECK: constraint of show at [[@LINE-2]]:22: methods.show(T) by the declaration's methods.show(T)
% CHECK: constraint of show at [[@LINE-3]]:33: methods.show(list.list(T)) by the instance at [[@LINE-5]]:13
% CHECK-NEXT: methods.show(T) by the declaration's methods.show(T)
    (show_to(Xs, S) :- S = show(Xs))
].

% A superclass's method on the instance's type, by its instance.
:- instance loud(int) where [ loud(X) = show(X) ++ "!" ].
% CHECK: clause at [[@LINE-1]]:31 of func methods.loud/1 in instance methods.loud(int): func methods.loud(int) = string
% CHECK: constraint of show at [[@LINE-2]]:41: methods.show(int) by the instance at 31:13

% The method's own constraint is assumed, on its own type variable, and
% the definition by name may be a method of another class.
:- instance conv(int, string) where [
    func(conv/1) is show,
% CHECK: clause at [[@LINE-1]]:5 of func methods.conv/1 in instance methods.conv(int, string): func methods.conv(int) = string
% CHECK: constraint of show at [[@LINE-2]]:21: methods.show(int) by the instance at 31:13
    with_shown(_, S, Y) = S ++ show(Y)
% CHECK: clause at [[@LINE-1]]:5 of func methods.with_shown/3 in instance methods.conv(int, string): func methods.with_shown(int, string, C) = string <= (methods.show(C))
% CHECK: constraint of show at [[@LINE-2]]:32: methods.show(C) by the declaration's methods.show(C)
].

% A loud(T) constraint assumes show(T), its superclass.
:- instance loud(list(T)) <= loud(T) where [
    loud([]) = "",
    loud([X | _]) = loud(X) ++ show([X])
% CHECK: constraint of show at [[@LINE-1]]:32: methods.show(list.list(T)) by the instance at 43:13
% CHECK-NEXT: methods.show(T) by the declaration's methods.loud(T)
].

% Calls of the methods choose these instances.
use = show([1, 2]) ++ loud(3) ++ conv(4) ++ with_shown(5, "", [6]).
% USE: constraint of with_shown at [[@LINE-1]]:45: methods.conv(int, string) by the instance at 60:13

% A function method may be defined by a constructor, as mmc allows (see
% the register).
:- typeclass boxed(T) where [ func boxed(T) = box(T) ].
:- type box(T) ---> box(T).
:- instance boxed(int) where [ func(boxed/1) is box ].
% CHECK: clause at [[@LINE-1]]:32 of func methods.boxed/1 in instance methods.boxed(int): func methods.boxed(int) = methods.box(int)
% CHECK-NEXT: [[@LINE-2]]:49 box -> constructor methods.box/1
