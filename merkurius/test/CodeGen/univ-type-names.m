% det_univ_to_type's message names the univ's type and the one asked for, as
% type_desc.type_name writes them (runtime/write.cpp, merk_type_name):
% qualified by their modules but for builtin types, tuples, and pred and func
% types; an equivalence type by what it stands for (docs/INPROGRESS.md,
% Exceptions, step 4). The last conversion is not caught. OUTPUT is mmc's,
% but for the header's "Merkurius".
% RUN: merkc -O0 %s -o %t0
% RUN: not %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: not %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: list.list(int)
% OUTPUT-NEXT:	Object Type: maybe.maybe(string)
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: univ_type_names.colour
% OUTPUT-NEXT:	Object Type: univ_type_names.box(float)
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: {int, character, string}
% OUTPUT-NEXT:	Object Type: pair.pair(int, string)
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: func(int) = int
% OUTPUT-NEXT:	Object Type: list.list(list.list(int))
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: array.array(int)
% OUTPUT-NEXT:	Object Type: list.list(univ_type_names.box(list.list(univ_type_names.colour)))
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: pred(int)
% OUTPUT-NEXT:	Object Type: character
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: {}
% OUTPUT-NEXT:	Object Type: float
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: int
% OUTPUT-NEXT:	Object Type: pred(io.state, io.state)
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: pred
% OUTPUT-NEXT:	Object Type: list.list(func(int) = int)
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: ((func) = int)
% OUTPUT-NEXT:	Object Type: list.list(((func) = int))
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: univ_type_names.t(int8, uint64)
% OUTPUT-NEXT:	Object Type: univ_type_names.t(bool.bool, int)
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: univ.univ
% OUTPUT-NEXT:	Object Type: univ_type_names.e
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: {{[{]}}{int}, list.list({})}
% OUTPUT-NEXT:	Object Type: uint
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: maybe.maybe(pred(int))
% OUTPUT-NEXT:	Object Type: string
% OUTPUT-NEXT:det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: uint8
% OUTPUT-NEXT:	Object Type: pair.pair(int32, float)
% OUTPUT-NEXT:ok
% OUTPUT-NEXT:Uncaught Merkurius exception:
% OUTPUT-NEXT:Software Error: det_univ_to_type: conversion failed
% OUTPUT-NEXT:	Univ Type: string
% OUTPUT-NEXT:	Object Type: int
% OUTPUT-NOT:{{.}}

:- module univ_type_names.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module array, bool, char, exception, float, int, list, maybe, pair,
    string, univ.

:- type colour ---> red ; green(int, string).
:- type box(T) ---> box(T).
:- type t(A, B) ---> t(A, B).
:- type e ---> e.
:- type ints == list(list(int)).

    % Converts the univ to the witness's type, and writes the message of the
    % software_error that det_univ_to_type throws.
:- pred convert(univ::in, T::in, io::di, io::uo) is cc_multi.

convert(Univ, Witness, !IO) :-
    try(
        ( pred(X::out) is det :-
            det_univ_to_type(Univ, X),
            _ = [X, Witness]
        ), Result),
    (
        Result = succeeded(_),
        io.write_string("ok\n", !IO)
    ;
        Result = exception(E),
        ( if univ_to_type(E, software_error(Message)) then
            io.write_string(Message ++ "\n", !IO)
        else
            io.write_string("another exception\n", !IO)
        )
    ).

main(!IO) :-
    convert(univ([1, 2]), yes("s"), !IO),
    convert(univ(red), box(1.5), !IO),
    convert(univ({1, 'c', "s"}), 1 - "x", !IO),
    convert(univ(func(X) = X + 1 : int), [[1]] : ints, !IO),
    convert(univ(array([1])), [box([red])], !IO),
    convert(univ((pred(Y::out) is det :- Y = 1) : pred(int)), 'c', !IO),
    convert(univ({}), 2.5, !IO),
    P2 = (pred(S0::di, S::uo) is det :- S = S0) : pred(io, io),
    convert(univ(1), P2, !IO),
    P0 = ((pred) is semidet :- true),
    F0 = ((func) = R :- R = 1),
    F1 = (func(A) = A + 1),
    convert(univ(P0), [F1], !IO),
    convert(univ(F0), [F0], !IO),
    convert(univ(t(1i8, 2u64)), t(yes, 0x41), !IO),
    convert(univ(univ(1)), e : e, !IO),
    convert(univ({{1}, [{}]}), 1u, !IO),
    convert(univ(yes(pred(_Z::in) is semidet :- true) : maybe(pred(int))),
        "s", !IO),
    convert(univ(1u8), 1i32 - 1.0, !IO),
    convert(univ(2), 3, !IO),
    % Uncaught.
    det_univ_to_type(univ("s"), N : int),
    io.write_int(N, !IO).
