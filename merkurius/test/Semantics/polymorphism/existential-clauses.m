% An existentially typed predicate with an existential constraint and
% several clauses: each clause gives the typeclass_info for the constraint
% on the type it chose, which all its clauses choose the same. mmc rejects
% it, as a mode error in the switch of its clauses (see the design
% document's mmc bugs), so it is not compared.
%
% RUN: merkc -fc1 -fdebug-dump-polymorphism %s 2>%t.err | FileCheck %s

:- module existential_clauses.
:- interface.
:- import_module list.
:- typeclass sh(T) where [ func sh(T) = int ].
:- instance sh(int).
:- instance sh(list(T)) <= sh(T).
:- type ab ---> a ; b.
:- some [T] pred multi(ab::in, T::out) is det => sh(T).
:- all [U] some [T] ((pred lists(ab::in, U::in, list(T)::out) is det
    => sh(T)) <= sh(U)).

:- implementation.
:- instance sh(int) where [ sh(X) = X ].
:- instance sh(list(T)) <= sh(T) where [ sh(_) = 0 ].

% CHECK-LABEL: pred existential_clauses.multi/2 mode 0:
% CHECK-NEXT: hidden arguments: existential_clauses.sh(T) (output)
% CHECK-NEXT: results: existential_clauses.sh(int) (constant)
% CHECK-NEXT: results: existential_clauses.sh(int) (constant)
multi(a, 1).
multi(b, 2).

% CHECK-LABEL: pred existential_clauses.lists/3 mode 0:
% CHECK-NEXT: hidden arguments: existential_clauses.sh(U), existential_clauses.sh(T) (output)
% CHECK-NEXT: type_infos in typeclass_infos: U (type_info 1 of existential_clauses.sh(U))
% CHECK-NEXT: results: existential_clauses.sh(U) (argument)
% CHECK-NEXT: results: existential_clauses.sh(U) (argument)
lists(a, U, [U]).
lists(b, _, []).
