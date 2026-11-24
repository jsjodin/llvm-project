% MANUAL: Existentially typed data types
% The manual's existentially typed data types: a heterogeneous list, one
% with a showable constraint built by make_list with 'new cons' and shown
% by process_list, the same through list(showable), and the type bar(T)
% with several quantified variables and constraints.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:["42", "1.0", "blah"]
% OUTPUT-NEXT:2
% OUTPUT-NEXT:["1", "x"]
% OUTPUT-NEXT:["f1", "f2 2", "f3", "4 four 4", "int and string"]
% OUTPUT-NOT:{{.}}

:- module run_showable.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, list, string.

    % A simple heterogeneous list type.
:- type list_of_any
    --->    nil_any
    ;       some [T] cons_any(T, list_of_any).

    % A heterogeneous list type with a type class constraint.
:- typeclass showable(T) where [ func show(T) = string ].
:- type showable_list
    --->    nil
    ;       some [T] (cons(T, showable_list) => showable(T)).

    % A different way of doing the same kind of thing, this
    % time using the standard type list(T).
:- type showable
    --->    some [T] (s(T) => showable(T)).
:- type list_of_showable == list(showable).

    % Here is an arbitrary example involving multiple type variables
    % and multiple constraints.
:- typeclass foo(T1, T2) where [ func pair_name(T1, T2) = string ].
:- type bar(T)
    --->    f1
    ;       f2(T)
    ;       some [T1] f3(T1)
    ;       some [T1, T2] f4(T1, T2, T) => (showable(T1), showable(T2))
    ;       some [T1, T2] f5(list(T1), T2) => foo(T1, T2).

:- instance showable(int) where [ show(I) = string.int_to_string(I) ].
:- instance showable(float) where [ show(F) = string.float_to_string(F) ].
:- instance showable(string) where [ show(S) = S ].
:- instance foo(int, string) where [ pair_name(_, _) = "int and string" ].

:- func make_list = showable_list.
make_list = List :-
    Int = 42,
    Float = 1.0,
    String = "blah",
    List =  'new cons'(Int,
        'new cons'(Float,
        'new cons'(String, nil))).

:- func process_list(showable_list) = list(string).
process_list(nil) = [].
process_list(cons(Head, Tail)) = [show(Head) | process_list(Tail)].

:- func count_any(list_of_any) = int.
count_any(nil_any) = 0.
count_any(cons_any(_, Tail)) = 1 + count_any(Tail).

:- func show_all(list_of_showable) = list(string).
show_all(L) = list.map((func(s(X)) = show(X)), L).

:- func describe(bar(int)) = string.
describe(f1) = "f1".
describe(f2(N)) = "f2 " ++ string.int_to_string(N).
describe(f3(_)) = "f3".
describe(f4(A, B, N)) = show(A) ++ " " ++ show(B) ++ " " ++
    string.int_to_string(N).
describe(f5(L, B)) = ( if L = [A | _] then pair_name(A, B) else "empty" ).

:- import_module int.

main(!IO) :-
    io.write_line(process_list(make_list), !IO),
    io.write_line(count_any('new cons_any'(1, 'new cons_any'("two", nil_any))),
        !IO),
    io.write_line(show_all(['new s'(1), 'new s'("x")]), !IO),
    io.write_line(list.map(describe, [f1, f2(2), 'new f3'(3.0),
        'new f4'(4, "four", 4), 'new f5'([5], "five")]), !IO).
