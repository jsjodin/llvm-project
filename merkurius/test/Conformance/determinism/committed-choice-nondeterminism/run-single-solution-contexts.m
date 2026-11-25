% MANUAL: Committed choice nondeterminism
% Calls to committed choice modes in single-solution contexts: the body of
% a cc_multi procedure, a goal without outputs, the last conjunct and those
% before a conjunct that cannot fail, an if-then-else's branches and its
% condition when the then part cannot fail. Two modes that differ only in
% cc-ness are chosen by the call's context.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:3
% OUTPUT-NEXT:{3, 4}
% OUTPUT-NEXT:"3"
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NOT:{{.}}

:- module run_single_solution_contexts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module int, list, solutions, string.

:- pred digit(int).
:- mode digit(out) is multi.
:- mode digit(out) is cc_multi.
digit(X) :-
    ( X = 3 ; X = 1 ; X = 2 ).

:- pred any_digit(int::out) is cc_multi.
any_digit(X) :-
    digit(X).

:- pred pair(int::out, int::out) is cc_multi.
pair(A, B) :-
    digit(A),
    B = A + 1.

:- pred chosen(string::out) is cc_multi.
chosen(S) :-
    ( if digit(D), D > 0 then
        S = string.int_to_string(D)
    else
        S = "none"
    ).

main(!IO) :-
    any_digit(D),
    io.write_line(D, !IO),
    pair(A, B),
    io.write_line({A, B}, !IO),
    chosen(S),
    io.write_line(S, !IO),
    solutions((pred(X::out) is multi :- digit(X)), Ds),
    io.write_line(Ds, !IO).
