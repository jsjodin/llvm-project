% MANUAL: The any inst
% A part of a term of inst any whose type cannot hold a solver type value
% is ground, and may be passed as in: an int, an enumeration, a type whose
% arguments are such, a list of ints, a string, a float, a closure, and a
% type variable's part where the type is known. A part that may hold one,
% a solver value in a list or in another type, or a type variable's, is
% not ground.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:49:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:51:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:53:{{[0-9]+}}: error:
:- module reject_any_parts_to_in.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, pair.

:- solver type st
    where representation is int,
    ground is ground,
    any is ground.

:- type colour ---> red ; green.
:- type ints ---> ints(int, int).
:- type wrap ---> wrap(st).
:- type msg(T) ---> fill(st, T, colour, ints, list(int), pair(int, string),
    float, pred(int), list(st), wrap).

:- pred use(T::in) is det.
use(_).

:- pred f_int(msg(int)::ia) is det.
f_int(fill(_, X, _, _, _, _, _, _, _, _)) :- use(X).
:- pred f_enum(msg(int)::ia) is det.
f_enum(fill(_, _, C, _, _, _, _, _, _, _)) :- use(C).
:- pred f_du(msg(int)::ia) is det.
f_du(fill(_, _, _, I, _, _, _, _, _, _)) :- use(I).
:- pred f_list(msg(int)::ia) is det.
f_list(fill(_, _, _, _, L, _, _, _, _, _)) :- use(L).
:- pred f_pair(msg(int)::ia) is det.
f_pair(fill(_, _, _, _, _, P, _, _, _, _)) :- use(P).
:- pred f_float(msg(int)::ia) is det.
f_float(fill(_, _, _, _, _, _, F, _, _, _)) :- use(F).
:- pred f_pred(msg(int)::ia) is det.
f_pred(fill(_, _, _, _, _, _, _, P, _, _)) :- use(P).
:- pred f_var(msg(T)::ia) is det.
f_var(fill(_, X, _, _, _, _, _, _, _, _)) :- use(X). % ERROR
:- pred f_list_st(msg(int)::ia) is det.
f_list_st(fill(_, _, _, _, _, _, _, _, L, _)) :- use(L). % ERROR
:- pred f_wrap(msg(int)::ia) is det.
f_wrap(fill(_, _, _, _, _, _, _, _, _, W)) :- use(W). % ERROR

main(!IO).
