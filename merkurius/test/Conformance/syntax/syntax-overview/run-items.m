% MANUAL: Syntax overview
% A module of declarations, facts, rules and a DCG rule, each item a
% term followed by a period.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:red[red]
% OUTPUT-NOT:{{.}}

:- module run_items.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type colour ---> red ; green.
:- pred is_red(colour::in) is semidet.
is_red(red).
:- func name(colour) = string.
name(red) = "red".
name(green) = Name :- Name = "green".
:- pred greens(list(colour)::in, list(colour)::out) is det.
greens --> ( [green] -> greens ; [] ).

main(!IO) :-
    ( if is_red(red) then io.write_string(name(red), !IO) else true ),
    greens([green, green, red], Rest),
    io.write(Rest, !IO), io.nl(!IO).
