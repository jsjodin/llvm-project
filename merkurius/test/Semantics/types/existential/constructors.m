% Existentially typed constructors (reference manual, "Existentially typed
% data types"): a construction, written 'new f', chooses the existential
% type variables, as a universally quantified function would, and must
% satisfy their constraints; a deconstruction, f without new, gives them
% types of its own, as an existentially typed function does, with their
% constraints. A field's selection is a deconstruction, and its update a
% construction of the field's existential variables. The first examples
% are the manual's.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types %s %}

:- module constructors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module float, int, list, string.

main(0).
:- typeclass showable(T) where [ func show(T) = string ].
:- instance showable(int) where [ show(X) = int_to_string(X) ].
:- instance showable(float) where [ show(X) = float_to_string(X) ].
:- instance showable(string) where [ show(S) = S ].

:- type showable_list
    --->    nil
    ;       some [T] (cons(T, showable_list) => showable(T)).

:- func make_list = showable_list.
make_list = List :-
    Int = 42,
    Float = 1.0,
    String = "blah",
    List = 'new cons'(Int, 'new cons'(Float, 'new cons'(String, nil))).
% CHECK: clause at [[@LINE-5]]:1 of func constructors.make_list/0: func constructors.make_list = constructors.showable_list
% CHECK: constraint of 'new cons' at [[@LINE-2]]:12: constructors.showable(int) by the instance at 21:13
% CHECK-NEXT: constraint of 'new cons' at [[@LINE-3]]:28: constructors.showable(float) by the instance at 22:13
% CHECK-NEXT: constraint of 'new cons' at [[@LINE-4]]:46: constructors.showable(string) by the instance at 23:13

:- func process_list(showable_list) = list(string).
process_list(nil) = [].
process_list(cons(Head, Tail)) = [show(Head) | process_list(Tail)].
% CHECK: clause at [[@LINE-1]]:1 of func constructors.process_list/1
% CHECK-NEXT: Head: (some [T] T)
% CHECK: constraint of show at [[@LINE-3]]:35: constructors.showable((some [T] T)) by the existential constraint of cons at [[@LINE-3]]:14

% Several variables and constraints, and a variable that a functional
% dependency determines.
:- typeclass foo(T1, T2) <= (T1 -> T2) where [].
:- instance foo(int, string) where [].
:- type bar(T)
    --->    f1
    ;       f2(T)
    ;       some [T1] f3(T1)
    ;       some [T1, T2] f4(T1, T2, T) => (showable(T1), showable(T2))
    ;       some [T1, T2] f5(list(T1), T2) => foo(T1, T2).
:- type determined ---> some [A, B] determined(A) => foo(A, B).

:- func bars = list(bar(float)).
bars = [f1, f2(1.5), 'new f3'("x"), 'new f4'(1, "y", 2.5), 'new f5'([1], "z"),
    constructors.'new f3'(1)].
% CHECK: clause at [[@LINE-2]]:1 of func constructors.bars/0: func constructors.bars = list.list(constructors.bar(float))
% CHECK: constraint of 'new f4' at [[@LINE-3]]:37: constructors.showable(int) by the instance at 21:13
% CHECK-NEXT: constraint of 'new f4' at [[@LINE-4]]:37: constructors.showable(string) by the instance at 23:13
% CHECK-NEXT: constraint of 'new f5' at [[@LINE-5]]:60: constructors.foo(int, string) by the instance at {{[0-9]+}}:13
:- func shown(bar(T)) = string.
shown(f1) = "".
shown(f2(_)) = "".
shown(f3(_)) = "".
shown(f4(A, B, _)) = show(A) ++ show(B).
% CHECK: constraint of show at [[@LINE-1]]:22: constructors.showable((some [T1] T1)) by the existential constraint of f4 at [[@LINE-1]]:7
% CHECK-NEXT: constraint of show at [[@LINE-2]]:33: constructors.showable((some [T2] T2)) by the existential constraint of f4 at [[@LINE-2]]:7
shown(f5(_, _)) = "".
:- func make_determined = determined.
make_determined = 'new determined'(1).
% CHECK: constraint of 'new determined' at [[@LINE-1]]:19: constructors.foo(int, string) by the instance at {{[0-9]+}}:13

% univ, with a type class constraint (reference manual, "Some idioms
% using existentially quantified types").
:- type univ_showable ---> some [T] (mkshowable(T) => showable(T)).
:- some [T] pred harder_example(string::in, T::out) is semidet => showable(T).
harder_example(Name, Showable) :-
    ( Name = "bar", Univ = 'new mkshowable'(42)
    ; Name = "foo", Univ = 'new mkshowable'("blah")
    ),
    Univ = mkshowable(Showable).
% CHECK: clause at [[@LINE-5]]:1 of pred constructors.harder_example/2: some [T] pred constructors.harder_example(string, T) => (constructors.showable(T))
% CHECK: existential constraint of the declaration: constructors.showable((some [T] T)) by the existential constraint of mkshowable at [[@LINE-2]]:12

% Fields: a selection gives the existential type, and an update chooses it,
% unless another field shares it.
:- type record ---> some [T] (record(item :: T, count :: int) => showable(T)).
:- func item_shown(record) = string.
item_shown(R) = show(R ^ item).
% CHECK: constraint of show at [[@LINE-1]]:17: constructors.showable((some [T] T)) by the existential constraint of item at [[@LINE-1]]:26
:- func recount(record) = record.
recount(R) = R ^ count := 2.
% CHECK: clause at [[@LINE-1]]:1 of func constructors.recount/1
% CHECK-NOT: constraint
% CHECK: clause at
:- func replaced(record) = record.
replaced(R) = R ^ item := 1.5.
% CHECK: [[@LINE-1]]:19 item -> field update constructors.'item :='/2
% CHECK-NEXT: constraint of item at [[@LINE-2]]:19: constructors.showable(float) by the instance at 22:13

% Inference: a deconstruction's existential type in the head is an
% existential variable, with its constraints; a construction's is not.
unboxed(mkshowable(X)) = X.
% CHECK: clause at [[@LINE-1]]:1 of func constructors.unboxed/1: inferred some [T] func constructors.unboxed(constructors.univ_showable) = T => (constructors.showable(T))
boxed(X) = 'new f3'(X).
% CHECK: clause at [[@LINE-1]]:1 of func constructors.boxed/1: inferred func constructors.boxed(T1) = constructors.bar(T2)
