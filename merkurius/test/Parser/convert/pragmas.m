% Pragmas, as the converter reads them, in a module that mmc compiles.
%
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck %s

:- module pragmas.
:- interface.

:- type colour ---> red ; green ; blue.
:- type handle.

:- func sin(float) = float.
:- pred double(int::in, int::out) is det.
:- func fib(int) = int.

:- implementation.
:- import_module int.

:- pragma foreign_decl("C", "#include <math.h>").
% CHECK:      | | ForeignDeclPragma
% CHECK-NEXT: | | | language: C
% CHECK-NEXT: | | | local: false
% CHECK-NEXT: | | | code: ForeignCode
% CHECK-NEXT: | | | | includeFile: false
% CHECK-NEXT: | | | | text: #include <math.h>
:- pragma foreign_decl("C", local, "static int calls = 0;").
% CHECK:      | | ForeignDeclPragma
% CHECK-NEXT: | | | language: C
% CHECK-NEXT: | | | local: true
% CHECK-NEXT: | | | code: ForeignCode
% CHECK-NEXT: | | | | includeFile: false
% CHECK-NEXT: | | | | text: static int calls = 0;
:- pragma foreign_code("C", "int pragmas_helper(int x) { return x; }").
% CHECK:      | | ForeignCodePragma
% CHECK-NEXT: | | | language: C
% CHECK-NEXT: | | | code: ForeignCode
% CHECK-NEXT: | | | | includeFile: false
% CHECK-NEXT: | | | | text: int pragmas_helper(int x) { return x; }

:- pragma foreign_type("C", handle, "void *", [can_pass_as_mercury_type]).
% CHECK:      | | ForeignTypePragma
% CHECK-NEXT: | | | language: C
% CHECK-NEXT: | | | name: SymName handle
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | descriptor: void *
% CHECK-NEXT: | | | assertions:
% CHECK-NEXT: | | | | CanPassAsMercuryType
% CHECK-NEXT: | | | attributes: none

:- pragma foreign_enum("C", colour/0, [red - "1", green - "2", blue - "3"]).
% CHECK:      | | ForeignEnumPragma
% CHECK-NEXT: | | | language: C
% CHECK-NEXT: | | | type: NameAndArity colour/0
% CHECK-NEXT: | | | values:
% CHECK-NEXT: | | | | ForeignEnumValue
% CHECK-NEXT: | | | | | constructor: SymName red
% CHECK-NEXT: | | | | | value: 1
% CHECK-NEXT: | | | | ForeignEnumValue
% CHECK-NEXT: | | | | | constructor: SymName green
% CHECK-NEXT: | | | | | value: 2
% CHECK-NEXT: | | | | ForeignEnumValue
% CHECK-NEXT: | | | | | constructor: SymName blue
% CHECK-NEXT: | | | | | value: 3
:- pragma foreign_export_enum("C", colour/0, [prefix("COLOUR_"), uppercase]).
% CHECK:      | | ForeignExportEnumPragma
% CHECK-NEXT: | | | language: C
% CHECK-NEXT: | | | type: NameAndArity colour/0
% CHECK-NEXT: | | | prefix: COLOUR_
% CHECK-NEXT: | | | uppercase: true
% CHECK-NEXT: | | | overrides:

:- pragma foreign_proc("C",
    sin(X::in) = (Sin::out),
    [promise_pure, will_not_call_mercury, thread_safe],
"
    Sin = sin(X);
").
% CHECK:      | | ForeignProcPragma
% CHECK-NEXT: | | | language: C
% CHECK-NEXT: | | | kind: Function
% CHECK-NEXT: | | | name: SymName sin
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | ForeignProcArg
% CHECK-NEXT: | | | | | var: Variable X
% CHECK-NEXT: | | | | | mode: NamedMode
% CHECK-NEXT: | | | | | | name: SymName in
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | result: ForeignProcArg
% CHECK-NEXT: | | | | var: Variable Sin
% CHECK-NEXT: | | | | mode: NamedMode
% CHECK-NEXT: | | | | | name: SymName out
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | attributes:
% CHECK-NEXT: | | | | PromisePure
% CHECK-NEXT: | | | | WillNotCallMercury
% CHECK-NEXT: | | | | ThreadSafe
% CHECK-NEXT: | | | code: ForeignCode
% CHECK-NEXT: | | | | includeFile: false
% CHECK-NEXT: | | | | text: "\n    Sin = sin(X);\n"

:- pragma inline(pred(double/2)).
% CHECK:      | | PredPragma
% CHECK-NEXT: | | | kind: Inline
% CHECK-NEXT: | | | pred: PredNameArity
% CHECK-NEXT: | | | | kind: Predicate
% CHECK-NEXT: | | | | nameArity: NameAndArity double/2
:- pragma foreign_export("C", double(in, out), "pragmas_double").
% CHECK:      | | ForeignExportPragma
% CHECK-NEXT: | | | language: C
% CHECK-NEXT: | | | proc: ProcSpec
% CHECK-NEXT: | | | | kind: Predicate
% CHECK-NEXT: | | | | name: SymName double
% CHECK-NEXT: | | | | modes:
% CHECK-NEXT: | | | | | NamedMode
% CHECK-NEXT: | | | | | | name: SymName in
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | NamedMode
% CHECK-NEXT: | | | | | | name: SymName out
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | result: none
% CHECK-NEXT: | | | foreignName: pragmas_double

double(X, Y) :-
    Y = X * 2.

:- pragma memo(fib/1, [allow_reset, statistics]).
% CHECK:      | | TablingPragma
% CHECK-NEXT: | | | method: Memo
% CHECK-NEXT: | | | pred: PredNameArity
% CHECK-NEXT: | | | | kind: none
% CHECK-NEXT: | | | | nameArity: NameAndArity fib/1
% CHECK-NEXT: | | | attributes: TablingAttributes
% CHECK-NEXT: | | | | allowReset: true
% CHECK-NEXT: | | | | statistics: true
% CHECK-NEXT: | | | | disableWarningIfIgnored: false
% CHECK-NEXT: | | | | fastLoose: false
% CHECK-NEXT: | | | | specified: none
% CHECK-NEXT: | | | | hiddenArg: none
% CHECK-NEXT: | | | | sizeLimit: none

fib(N) = F :-
    ( if N < 2 then
        F = N
    else
        F = fib(N - 1) + fib(N - 2)
    ).

:- pred old(int::in, int::out) is det.
:- pragma obsolete(old/2, [double/2]).
% CHECK:      | | ObsoletePragma
% CHECK-NEXT: | | | pred: PredNameArity
% CHECK-NEXT: | | | | kind: none
% CHECK-NEXT: | | | | nameArity: NameAndArity old/2
% CHECK-NEXT: | | | replacements:
% CHECK-NEXT: | | | | NameAndArity double/2
:- pragma consider_used(pred(old/2)).
% CHECK:      | | PredPragma
% CHECK-NEXT: | | | kind: ConsiderUsed
% CHECK-NEXT: | | | pred: PredNameArity
% CHECK-NEXT: | | | | kind: Predicate
% CHECK-NEXT: | | | | nameArity: NameAndArity old/2
:- pragma no_determinism_warning(old/2).
% CHECK:      | | PredPragma
% CHECK-NEXT: | | | kind: NoDeterminismWarning
% CHECK-NEXT: | | | pred: PredNameArity
% CHECK-NEXT: | | | | kind: none
% CHECK-NEXT: | | | | nameArity: NameAndArity old/2

old(X, Y) :-
    double(X, Y).

:- pred len(list(T)::in, int::out) is det.
:- pragma type_spec(len/2, T = int).
% CHECK:      | | TypeSpecPragma
% CHECK-NEXT: | | | pred: PredNameArity
% CHECK-NEXT: | | | | kind: none
% CHECK-NEXT: | | | | nameArity: NameAndArity len/2
% CHECK-NEXT: | | | substitution:
% CHECK-NEXT: | | | | TypeSubstitution
% CHECK-NEXT: | | | | | var: T
% CHECK-NEXT: | | | | | type: NamedType
% CHECK-NEXT: | | | | | | name: SymName int
% CHECK-NEXT: | | | | | | arguments:
:- import_module list.

len([], 0).
len([_ | Xs], N + 1) :-
    len(Xs, N).
