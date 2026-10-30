% The names in declarations: types, insts, modes and type classes, the
% type of an inst, a class constraint, an instance's class, the modes of a
% lambda, and a type qualification. A class method is a function or
% predicate of the module. (Instance methods' clauses are not desugared yet,
% so their names are not resolved.)
%
% RUN: merkc -fc1 -fdebug-dump-resolved-names %s 2>&1 | FileCheck --implicit-check-not=error %s

:- module declarations.
:- interface.
:- import_module list.
:- typeclass shape(T) where [
    func area(T) = float
].
:- type square ---> square(float).
:- instance shape(square).
:- type shapes(T) == list(T).
:- inst nonempty for list/1 ---> [ground | ground].
:- mode in_nonempty == in(nonempty).
:- pred total(shapes(T)::in, float::out) is det <= shape(T).
:- pred first(list(int)::in_nonempty, int::out) is det.
:- implementation.
:- import_module float.
:- instance shape(square) where [
    area(square(S)) = S * S
].
total(Shapes, Total) :-
    Add = (pred(S::in, T0::in, T::out) is det :- T = T0 + area(S)),
    list.foldl(Add, Shapes, 0.0, Total).
first([X | _], X : int).

% CHECK:      symbols of declarations:
% CHECK-NEXT:   12:14 typeclass declarations.shape/1 exported
% CHECK-NEXT:   13:10 func declarations.area/1 exported of typeclass declarations.shape/1
% CHECK-NEXT:   15:9 type declarations.square/0 exported
% CHECK-NEXT:   15:21 constructor declarations.square/1 exported of type declarations.square/0
% CHECK-NEXT:   17:9 type declarations.shapes/1 exported
% CHECK-NEXT:   18:9 inst declarations.nonempty/0 exported
% CHECK-NEXT:   19:9 mode declarations.in_nonempty/0 exported
% CHECK-NEXT:   20:9 pred declarations.total/2 exported
% CHECK-NEXT:   21:9 pred declarations.first/2 exported
% CHECK-NEXT: names:
% CHECK-NEXT:   13:20 float Type/0 -> type float/0
% CHECK-NEXT:   15:28 float Type/0 -> type float/0
% CHECK-NEXT:   16:13 shape Typeclass/1 -> typeclass declarations.shape/1
% CHECK-NEXT:   16:19 square Type/0 -> type declarations.square/0
% CHECK-NEXT:   17:22 list Type/1 -> type list.list/1
% CHECK-NEXT:   18:22 list Type/1 -> type list.list/1
% CHECK-NEXT:   19:27 nonempty Inst/0 -> inst declarations.nonempty/0
% CHECK-NEXT:   19:24 in Mode/1 -> mode builtin.in/1
% CHECK-NEXT:   20:15 shapes Type/1 -> type declarations.shapes/1
% CHECK-NEXT:   20:26 in Mode/0 -> mode builtin.in/0
% CHECK-NEXT:   20:30 float Type/0 -> type float/0
% CHECK-NEXT:   20:37 out Mode/0 -> mode builtin.out/0
% CHECK-NEXT:   20:52 shape Typeclass/1 -> typeclass declarations.shape/1
% CHECK-NEXT:   21:20 int Type/0 -> type int/0
% CHECK-NEXT:   21:15 list Type/1 -> type list.list/1
% CHECK-NEXT:   21:26 in_nonempty Mode/0 -> mode declarations.in_nonempty/0
% CHECK-NEXT:   21:39 int Type/0 -> type int/0
% CHECK-NEXT:   21:44 out Mode/0 -> mode builtin.out/0
% CHECK-NEXT:   24:13 shape Typeclass/1 -> typeclass declarations.shape/1
% CHECK-NEXT:   24:19 square Type/0 -> type declarations.square/0
% CHECK-NEXT:   27:1 total Definition/2 -> pred declarations.total/2
% CHECK-NEXT:   28:20 in Mode/0 -> mode builtin.in/0
% CHECK-NEXT:   28:28 in Mode/0 -> mode builtin.in/0
% CHECK-NEXT:   28:35 out Mode/0 -> mode builtin.out/0
% CHECK-NEXT:   28:57 + DataTerm/2 -> func float.'+'/2
% CHECK-NEXT:   28:59 area DataTerm/1 -> func declarations.area/1
% CHECK-NEXT:   29:5 list.foldl Call/4 -> pred list.foldl/4
% CHECK-NEXT:   30:1 first Definition/2 -> pred declarations.first/2
% CHECK-NEXT:   30:20 int Type/0 -> type int/0
