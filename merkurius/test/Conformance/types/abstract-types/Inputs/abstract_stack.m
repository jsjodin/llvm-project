% An abstract type for the tests of Abstract types: stack/1 is a
% discriminated union and depth/0 an equivalence type, both hidden.

:- module abstract_stack.
:- interface.

:- type stack(T).
:- type depth.

:- func empty = stack(T).
:- func push(T, stack(T)) = stack(T).
:- pred pop(stack(T)::in, T::out, stack(T)::out) is semidet.
:- func depth(stack(T)) = depth.
:- func depth_to_int(depth) = int.

:- implementation.
:- import_module int.

:- type stack(T)
    --->    nil
    ;       cons(T, stack(T)).

:- type depth == int.

empty = nil.
push(X, S) = cons(X, S).
pop(cons(X, S), X, S).
depth(nil) = 0.
depth(cons(_, S)) = 1 + depth(S).
depth_to_int(D) = D.
