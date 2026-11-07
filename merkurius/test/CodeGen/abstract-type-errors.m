% A type exported abstractly must pass as a pointer does, since other
% modules, which cannot see its layout, pass its values as pointers: one of
% cells or an enum, or a notag of a value that is a pointer or a word. A
% notag of a float, or of a narrower integer or a character, directly or
% through another notag, cannot yet. A type not exported, or exported with
% its constructors, may have any layout.
%
% RUN: not merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module abstract_type_errors.
:- interface.
:- import_module char, int8.

:- type celsius.
:- type small.
:- type letter.
:- type kelvin.
:- type count.
:- type label.
:- type colour.
:- type cells.
:- type shown ---> shown(float).

:- func make_celsius(float) = celsius.
:- func make_small(int8) = small.
:- func make_letter(char) = letter.
:- func make_kelvin(float) = kelvin.
:- func make_count(int) = count.
:- func make_label(string) = label.
:- func make_red = colour.
:- func make_cells(int) = cells.
:- func hidden(float) = float.

:- implementation.

:- type celsius ---> celsius(float).
% CHECK-DAG: abstract-type-errors.m:[[@LINE-1]]:9: error: not yet implemented: a type exported abstractly whose values are f64, which other modules pass as pointers
:- type small ---> small(int8).
% CHECK-DAG: abstract-type-errors.m:[[@LINE-1]]:9: error: not yet implemented: a type exported abstractly whose values are i8, which other modules pass as pointers
:- type letter ---> letter(char).
% CHECK-DAG: abstract-type-errors.m:[[@LINE-1]]:9: error: not yet implemented: a type exported abstractly whose values are i32, which other modules pass as pointers
:- type kelvin ---> kelvin(degrees).
% CHECK-DAG: abstract-type-errors.m:[[@LINE-1]]:9: error: not yet implemented: a type exported abstractly whose values are f64, which other modules pass as pointers
:- type degrees ---> degrees(float).
:- type count ---> count(int).
:- type label ---> label(string).
:- type colour ---> crimson ; navy.
:- type cells ---> one(int) ; two(int, int).
:- type secret ---> secret(float).

make_celsius(F) = celsius(F).
make_small(N) = small(N).
make_letter(C) = letter(C).
make_kelvin(F) = kelvin(degrees(F)).
make_count(N) = count(N).
make_label(S) = label(S).
make_red = crimson.
make_cells(N) = one(N).
hidden(F) = G :-
    S = secret(F),
    S = secret(G),
    W = shown(G),
    W = shown(_).
