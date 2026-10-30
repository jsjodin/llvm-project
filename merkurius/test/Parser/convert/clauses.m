% Clauses and the goals and data-terms in them, as the converter reads
% them, in a module that mmc compiles. DCG rules, state variables and
% field access stay as written.
%
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck %s

:- module clauses.
:- interface.
:- import_module io.

:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module char, int, list, stm_builtin, string.

:- type point ---> point(x :: int, y :: int).

main(!IO) :-
    io.write_string("hello\n", !IO),
    count([1, 2, 3], N),
    ( if N > 2 then
        io.write_int(N, !IO)
    else
        true
    ),
    P = point(1, 2),
    Q = P ^ x := 3,
    io.write_int(Q ^ x + Q ^ y, !IO),
    Add = (func(A) = A + N),
    io.write_int(Add(1), !IO),
    trace [compile_time(flag("debug")), io(!TIO)] (
        io.write_string("traced\n", !TIO)
    ).
% CHECK:      | | FactOrRule
% CHECK-NEXT: | | | kind: Predicate
% CHECK-NEXT: | | | name: SymName main
% CHECK:      | | | | | Call
% CHECK:      | | | | | Call
% CHECK:      | | | | | IfThenElse
% CHECK:      | | | | | | condition: Call
% CHECK:      | | | | | | then: Call
% CHECK:      | | | | | | else: BuiltinGoal
% CHECK:      | | | | | Unification
% CHECK:      | | | | | | right: FunctorData
% CHECK:      | | | | | Unification
% CHECK:      | | | | | | right: FieldUpdate
% CHECK:      | | | | | Call
% CHECK:      | | | | | Unification
% CHECK:      | | | | | | right: Lambda
% CHECK:      | | | | | Call
% CHECK:      | | | | | TraceGoal
% CHECK:      | | | | | | | | condition: TraceCondition
% CHECK-NEXT: | | | | | | | | | kind: Flag
% CHECK-NEXT: | | | | | | | | | operands:
% CHECK-NEXT: | | | | | | | | | value: debug
% CHECK:      | | | | | | | | condition: none

% A goal that updates a field of a state variable.
:- pred move(point::in, point::out) is det.

move(!P) :-
    !P ^ x := 0,
    !P ^ y := !.P ^ x + 1.
% CHECK:      | | FactOrRule
% CHECK-NEXT: | | | kind: Predicate
% CHECK-NEXT: | | | name: SymName move
% CHECK:      | | | | | StateVarFieldUpdate
% CHECK-NEXT: | | | | | | var: StateVar !P
% CHECK-NEXT: | | | | | | fields:
% CHECK-NEXT: | | | | | | | FieldSpec
% CHECK-NEXT: | | | | | | | | name: SymName x
% CHECK-NEXT: | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | value: Literal 0
% CHECK:      | | | | | StateVarFieldUpdate
% CHECK-NEXT: | | | | | | var: StateVar !P
% CHECK:      | | | | | | value: FunctorData

% An atomic goal, a transaction whose alternatives are taken in turn.
:- pred choose(int::out, io::di, io::uo) is cc_multi.

choose(X, !IO) :-
    atomic [outer(!IO), inner(!STM), vars([X])] (
        X = 1
    or_else
        ( X = 2 or_else X = 3 )
    ).
% CHECK:      | | FactOrRule
% CHECK-NEXT: | | | kind: Predicate
% CHECK-NEXT: | | | name: SymName choose
% CHECK:      | | | body: AtomicGoal
% CHECK-NEXT: | | | | params:
% CHECK-NEXT: | | | | | AtomicParam
% CHECK-NEXT: | | | | | | kind: Outer
% CHECK-NEXT: | | | | | | initial: StateVar !IO
% CHECK-NEXT: | | | | | | final: none
% CHECK-NEXT: | | | | | | vars:
% CHECK-NEXT: | | | | | AtomicParam
% CHECK-NEXT: | | | | | | kind: Inner
% CHECK-NEXT: | | | | | | initial: StateVar !STM
% CHECK-NEXT: | | | | | | final: none
% CHECK-NEXT: | | | | | | vars:
% CHECK-NEXT: | | | | | AtomicParam
% CHECK-NEXT: | | | | | | kind: Vars
% CHECK-NEXT: | | | | | | initial: none
% CHECK-NEXT: | | | | | | final: none
% CHECK-NEXT: | | | | | | vars:
% CHECK-NEXT: | | | | | | | X
% The or_else alternatives flatten through parentheses.
% CHECK-NEXT: | | | | goals:
% CHECK-NEXT: | | | | | Unification
% CHECK-NEXT: | | | | | | left: Variable X
% CHECK-NEXT: | | | | | | right: Literal 1
% CHECK-NEXT: | | | | | Unification
% CHECK-NEXT: | | | | | | left: Variable X
% CHECK-NEXT: | | | | | | right: Literal 2
% CHECK-NEXT: | | | | | Unification
% CHECK-NEXT: | | | | | | left: Variable X
% CHECK-NEXT: | | | | | | right: Literal 3

:- pred count(list(T)::in, int::out) is det.

count([], 0).
% CHECK:      | | FactOrRule
% CHECK-NEXT: | | | kind: Predicate
% CHECK-NEXT: | | | name: SymName count
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | ModedTerm
% CHECK-NEXT: | | | | | term: ListData
% CHECK-NEXT: | | | | | | elements:
% CHECK-NEXT: | | | | | | tail: none
% CHECK-NEXT: | | | | | mode: none
% CHECK-NEXT: | | | | ModedTerm
% CHECK-NEXT: | | | | | term: Literal 0
% CHECK-NEXT: | | | | | mode: none
% CHECK-NEXT: | | | result: none
% CHECK-NEXT: | | | body: none
count([_ | Xs], N + 1) :-
    count(Xs, N).
% CHECK:      | | FactOrRule
% CHECK-NEXT: | | | kind: Predicate
% CHECK-NEXT: | | | name: SymName count
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | ModedTerm
% CHECK-NEXT: | | | | | term: ListData
% CHECK-NEXT: | | | | | | elements:
% CHECK-NEXT: | | | | | | | Variable _
% CHECK-NEXT: | | | | | | tail: Variable Xs
% CHECK-NEXT: | | | | | mode: none
% CHECK-NEXT: | | | | ModedTerm
% CHECK-NEXT: | | | | | term: FunctorData
% CHECK-NEXT: | | | | | | name: SymName +
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | | | Variable N
% CHECK-NEXT: | | | | | | | Literal 1
% CHECK-NEXT: | | | | | mode: none
% CHECK-NEXT: | | | result: none
% CHECK-NEXT: | | | body: Call
% CHECK-NEXT: | | | | name: SymName count
% CHECK-NEXT: | | | | arguments:
% CHECK-NEXT: | | | | | Variable Xs
% CHECK-NEXT: | | | | | Variable N

:- func double(int) = int.

double(X) = Y :-
    Y = X * 2.
% CHECK:      | | FactOrRule
% CHECK-NEXT: | | | kind: Function
% CHECK-NEXT: | | | name: SymName double
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | ModedTerm
% CHECK-NEXT: | | | | | term: Variable X
% CHECK-NEXT: | | | | | mode: none
% CHECK-NEXT: | | | result: ModedTerm
% CHECK-NEXT: | | | | term: Variable Y
% CHECK-NEXT: | | | | mode: none
% CHECK-NEXT: | | | body: Unification
% CHECK-NEXT: | | | | left: Variable Y
% CHECK-NEXT: | | | | right: FunctorData
% CHECK-NEXT: | | | | | name: SymName *
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | | | | Variable X
% CHECK-NEXT: | | | | | | Literal 2

:- pred digits(list(char)::out, list(char)::in, list(char)::out) is multi.

digits([D | Ds]) -->
    [D],
    { char.is_digit(D) },
    digits(Ds).
% CHECK:      | | DcgRule
% CHECK-NEXT: | | | name: SymName digits
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | ModedTerm
% CHECK-NEXT: | | | | | term: ListData
% CHECK-NEXT: | | | | | | elements:
% CHECK-NEXT: | | | | | | | Variable D
% CHECK-NEXT: | | | | | | tail: Variable Ds
% CHECK-NEXT: | | | | | mode: none
% CHECK-NEXT: | | | body: DcgConjunction
% CHECK-NEXT: | | | | parallel: false
% CHECK-NEXT: | | | | goals:
% CHECK-NEXT: | | | | | DcgTerminals
% CHECK-NEXT: | | | | | | terms:
% CHECK-NEXT: | | | | | | | Variable D
% CHECK-NEXT: | | | | | DcgBraces
% CHECK-NEXT: | | | | | | goal: Call
% CHECK-NEXT: | | | | | | | name: SymName char.is_digit
% CHECK-NEXT: | | | | | | | arguments:
% CHECK-NEXT: | | | | | | | | Variable D
% CHECK-NEXT: | | | | | DcgCall
% CHECK-NEXT: | | | | | | name: SymName digits
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | | | Variable Ds
digits([]) --> [].
% CHECK:      | | DcgRule
% CHECK-NEXT: | | | name: SymName digits
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | ModedTerm
% CHECK-NEXT: | | | | | term: ListData
% CHECK-NEXT: | | | | | | elements:
% CHECK-NEXT: | | | | | | tail: none
% CHECK-NEXT: | | | | | mode: none
% CHECK-NEXT: | | | body: DcgTerminals
% CHECK-NEXT: | | | | terms:
