% Code generation reports the first construct of each procedure that it
% does not handle yet, and fails.
%
% RUN: not merkc -fc1 -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module not_yet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, exception, int, list, string.

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
:- func joined(string) = string.
joined(S) = S ++ "!".
% CHECK: not-yet.m:[[@LINE-1]]:13: error: not yet implemented: code generation for calls of string.++/2

:- func upper(char) = char.
upper(C) = char.to_upper(C).
% CHECK: not-yet.m:[[@LINE-1]]:12: error: not yet implemented: code generation for calls of char.to_upper/1

:- pred thrown(int::out) is det.
thrown(X) :-
    X = 1,
    throw(X).
% CHECK: not-yet.m:[[@LINE-1]]:5: error: not yet implemented: code generation for calls of exception.throw/1
