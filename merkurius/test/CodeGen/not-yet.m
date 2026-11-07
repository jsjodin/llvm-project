% Code generation reports the first construct of each procedure that it
% does not handle yet, and fails.
%
% RUN: not merkc -fc1 -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module not_yet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, bool, exception, int, list.

main(!IO).

:- typeclass size(T) where [ func size(T) = int ].
:- instance size(int) where [ size(X) = X ].
:- func sizes(list(int)) = list(func(int) = int).
sizes(L) = [size | map_size(L)].
% CHECK: not-yet.m:[[@LINE-1]]:13: error: not yet implemented: code generation for closures of methods
:- func map_size(list(int)) = list(func(int) = int).
map_size(_) = [].

% A type with user-defined equality, whose unifications call its
% predicate.
:- type set ---> set(list(int)) where equality is equal.
:- pred equal(set::in, set::in) is semidet.
% CHECK: not-yet.m:[[@LINE-1]]:9: error: not yet implemented: code generation for values of type not_yet.set
equal(set(X), set(X)).

:- some [T] pred r(T::out) is det.
% CHECK: not-yet.m:[[@LINE-1]]:18: error: not yet implemented: code generation for foreign_procs with existential types
:- pragma foreign_proc("C", r(X::out), [will_not_call_mercury, promise_pure], "X = 0;").

:- func w(int) = int.
w(X) = pow(X, 3).
% CHECK: not-yet.m:[[@LINE-1]]:8: error: not yet implemented: code generation for calls of int.pow/2

% Calls of library modules that are not compiled yet and not built in.
:- func negate(bool) = bool.
negate(B) = bool.not(B).
% CHECK: not-yet.m:[[@LINE-1]]:13: error: not yet implemented: code generation for calls of bool.not/1

% Of array, only generate, its folds and the foldl family are compiled.
:- func total(array(int)) = int.
total(A) = array.foldr((func(X, S) = X + S), A, 0).
% CHECK: not-yet.m:[[@LINE-1]]:12: error: not yet implemented: code generation for calls of array.foldr/3

:- pred thrown(int::out) is det.
thrown(X) :-
    X = 1,
    throw(X).
% CHECK: not-yet.m:[[@LINE-1]]:5: error: not yet implemented: code generation for calls of exception.throw/1
