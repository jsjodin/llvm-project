% Solver types, initialise, finalise, mutable and promise declarations, as
% the converter reads them, in a module that mmc compiles.
%
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck %s

:- module directives.
:- interface.
:- import_module io.

:- solver type var.
% CHECK:      | | AbstractTypeDecl
% CHECK-NEXT: | | | name: SymName var
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | attributes: none
% CHECK-NEXT: | | | solver: true

:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module int.

:- solver type var
    where representation is int,
          ground is ground,
          any is ground,
          constraint_store is mutable(next_var, int, 0, ground, [untrailed]).
% CHECK:      | | SolverTypeDefn
% CHECK-NEXT: | | | name: SymName var
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | representation: NamedType
% CHECK-NEXT: | | | | name: SymName int
% CHECK-NEXT: | | | | arguments:
% CHECK-NEXT: | | | ground: BuiltinInst
% CHECK-NEXT: | | | | kind: Ground
% CHECK-NEXT: | | | any: BuiltinInst
% CHECK-NEXT: | | | | kind: Ground
% CHECK-NEXT: | | | constraintStore:
% CHECK-NEXT: | | | | MutableDecl
% CHECK-NEXT: | | | | | name: next_var
% CHECK-NEXT: | | | | | type: NamedType
% CHECK-NEXT: | | | | | | name: SymName int
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | initial: Literal 0
% CHECK-NEXT: | | | | | inst: BuiltinInst
% CHECK-NEXT: | | | | | | kind: Ground
% CHECK-NEXT: | | | | | trailed: false
% CHECK-NEXT: | | | | | attachToIoState: false
% CHECK-NEXT: | | | | | constant: false
% CHECK-NEXT: | | | | | threadLocal: false
% CHECK-NEXT: | | | | | foreignNames:
% CHECK-NEXT: | | | equality: none
% CHECK-NEXT: | | | comparison: none

:- initialise init/2.
% CHECK:      | | InitialiseDecl
% CHECK-NEXT: | | | kind: Initialise
% CHECK-NEXT: | | | pred: NameAndArity init/2
:- finalise fin/2.
% CHECK:      | | InitialiseDecl
% CHECK-NEXT: | | | kind: Finalise
% CHECK-NEXT: | | | pred: NameAndArity fin/2

:- mutable(counter, int, 0, ground, [untrailed, attach_to_io_state]).
% CHECK:      | | MutableDecl
% CHECK-NEXT: | | | name: counter
% CHECK-NEXT: | | | type: NamedType
% CHECK-NEXT: | | | | name: SymName int
% CHECK-NEXT: | | | | arguments:
% CHECK-NEXT: | | | initial: Literal 0
% CHECK-NEXT: | | | inst: BuiltinInst
% CHECK-NEXT: | | | | kind: Ground
% CHECK-NEXT: | | | trailed: false
% CHECK-NEXT: | | | attachToIoState: true
% CHECK-NEXT: | | | constant: false
% CHECK-NEXT: | | | threadLocal: false
% CHECK-NEXT: | | | foreignNames:

:- pred init(io::di, io::uo) is det.
init(!IO) :-
    set_counter(1, !IO).

:- pred fin(io::di, io::uo) is det.
fin(!IO) :-
    get_counter(N, !IO),
    io.write_int(N, !IO).

main(!IO) :-
    io.write_string("main\n", !IO).

:- pred colour(int::in) is semidet.
colour(1).
colour(2).

:- promise all [X] (colour(X) => X > 0).
% CHECK:      | | PromiseDecl
% CHECK-NEXT: | | | goal: QuantifiedGoal
% CHECK-NEXT: | | | | quantifier: All
% CHECK-NEXT: | | | | vars:
% CHECK-NEXT: | | | | | Variable X
% CHECK-NEXT: | | | | goal: Implication
% CHECK-NEXT: | | | | | kind: Implies
% CHECK-NEXT: | | | | | left: Call
% CHECK-NEXT: | | | | | | name: SymName colour
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | | | Variable X
% CHECK-NEXT: | | | | | right: Call
% CHECK-NEXT: | | | | | | name: SymName >
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | | | Variable X
% CHECK-NEXT: | | | | | | | Literal 0
:- all [X] promise_exclusive (X = 1 ; X = 2).
% CHECK:      | | PromiseDisjunctionDecl
% CHECK-NEXT: | | | kind: Exclusive
% CHECK-NEXT: | | | vars:
% CHECK-NEXT: | | | | X
% CHECK-NEXT: | | | disjunction: Disjunction
% CHECK-NEXT: | | | | goals:
% CHECK-NEXT: | | | | | Unification
% CHECK-NEXT: | | | | | | left: Variable X
% CHECK-NEXT: | | | | | | right: Literal 1
% CHECK-NEXT: | | | | | Unification
% CHECK-NEXT: | | | | | | left: Variable X
% CHECK-NEXT: | | | | | | right: Literal 2
