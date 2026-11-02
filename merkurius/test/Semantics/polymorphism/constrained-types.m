% A type variable whose first constraint has it only inside a type, as in
% sh(list(T)), has as its type_info that type's argument, through as many
% types as it takes: the type_info in the typeclass_info is list(T)'s. It
% is the first constraint that has it, even if a later one has it itself,
% as in mmc. mmc passes list(T)'s as T's, so that
% type_of(X) is list.list(int) for an X of type int (see the register);
% merkc does not follow it, and so is not compared with mmc here.
%
% RUN: merkc -fc1 -fdebug-dump-polymorphism %s 2>%t.err | FileCheck %s

:- module constrained_types.
:- interface.
:- import_module list, pair.
:- typeclass sh(T) where [ func sh(T) = int ].
:- instance sh(list(T)).
:- instance sh(pair(A, B)).
:- pred cons(T::in, int::out) is det <= sh(list(T)).
% CHECK-LABEL: pred constrained_types.cons/2 mode 0:
% CHECK-NEXT: hidden arguments: constrained_types.sh(list.list(T))
% CHECK-NEXT: type_infos in typeclass_infos: T (argument 1 of list.list(T) (type_info 1 of constrained_types.sh(list.list(T))))
% CHECK-NEXT: call pred constrained_types.unused/2: T (argument 1 of list.list(T) (type_info 1 of constrained_types.sh(list.list(T))))
:- pred first(T::in, list(U)::in, int::out) is det <=
    (sh(pair(list(U), T)), sh(T)).
% CHECK-LABEL: pred constrained_types.first/3 mode 0:
% CHECK-NEXT: hidden arguments: constrained_types.sh(pair.pair(list.list(U), T)), constrained_types.sh(T)
% CHECK-NEXT: type_infos in typeclass_infos: T (argument 2 of pair.pair(list.list(U), T) (type_info 1 of constrained_types.sh(pair.pair(list.list(U), T)))), U (argument 1 of list.list(U) (argument 1 of pair.pair(list.list(U), T) (type_info 1 of constrained_types.sh(pair.pair(list.list(U), T)))))
% CHECK-NEXT: call pred constrained_types.unused/2: T (argument 2 of pair.pair(list.list(U), T) (type_info 1 of constrained_types.sh(pair.pair(list.list(U), T))))
:- pred unused(T::in, int::out) is det.
:- implementation.
:- instance sh(list(T)) where [ sh(_) = 0 ].
:- instance sh(pair(A, B)) where [ sh(_) = 1 ].
unused(_, 1).
cons(X, N) :- unused(X, N).
first(X, _, N) :- unused(X, N).
