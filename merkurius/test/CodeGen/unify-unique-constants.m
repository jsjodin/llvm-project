% A unification of two values that are both the constant kk, still unique,
% of a type that also has a constructor with arguments, is a test that
% succeeds. mmc 22.01.9 rejects it, with a mode error in its own __Unify__
% for t (design document, the register, "A unification of two unique
% constants").
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --match-full-lines --strict-whitespace

:- module unify_unique_constants.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- type t
    --->    c(int)
    ;       kk.

main(!IO) :-
    S = kk,
    D = kk,
    ( if S = D then io.write_string("eq\n", !IO) else io.write_string("ne\n", !IO) ).
% CHECK:eq
