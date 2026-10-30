% Equivalence types are expanded, so a type and its equivalent are one
% type: char.char is the primitive character, io.io is io.state, and builtin's
% comparison_pred(T) is a pred type. A declaration can give its type as an
% equivalence for a higher-order type, which adds that type's arguments.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error --implicit-check-not=warning %s

:- module equivalences.
:- interface.
:- import_module char, io, list.
:- type name == string.
:- type pairs(T) == list({name, T}).
:- type folder(T) == pred(T, int, int).
:- pred first(pairs(char)::in, name::out) is semidet.
:- pred count : folder(T) `with_inst` (pred(in, in, out) is det).
:- pred greet(io::di, io::uo) is det.
:- implementation.
:- import_module int.

first([{N, C} | _], N) :- char.is_upper(C).
count(_, N0, N) :- N = N0 + 1.
greet(!IO) :- io.write_string("hi", !IO).

% CHECK:      clause at 20:1 of pred equivalences.first/2: pred equivalences.first(list.list({string, character}), string)
% CHECK-NEXT:   N: string
% CHECK-NEXT:   C: character
% CHECK:      clause at 21:1 of pred equivalences.count/3: pred equivalences.count(T, int, int)
% CHECK-NEXT:   N0: int
% CHECK-NEXT:   N: int
% CHECK:      clause at 22:1 of pred equivalences.greet/2: pred equivalences.greet(io.state, io.state)
