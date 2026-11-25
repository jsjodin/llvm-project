% MANUAL: Examples of correct code using type quantifiers
% The manual's examples of correct code: universal and existential
% predicates and their callers, and the higher-order ones, both of which
% return 13.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:13
% OUTPUT-NEXT:13
% OUTPUT-NOT:{{.}}

:- module run_manual_examples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

/* simple examples */

:- pred foo(T).
:- mode foo(in) is det.
foo(_).
    % ok

:- pred call_foo.
:- mode call_foo is det.
call_foo :- foo(42).
    % ok (T = int)

:- some [T] pred e_foo(T).
:- mode e_foo(out) is det.
e_foo(X) :- X = 42.
    % ok (T = int)

:- pred call_e_foo.
:- mode call_e_foo is det.
call_e_foo :- e_foo(_).
    % ok

/* examples using higher-order functions */

:- func bar(T, T, func(T) = int) = int.
bar(X, Y, F) = F(X) + F(Y).
    % ok

:- func call_bar = int.
call_bar = bar(2, 3, (func(X) = X*X)).
    % ok (T = int)
    % returns 13 (= 2*2 + 3*3)

:- some [T] pred e_bar(T, T, func(T) = int).
:-          mode e_bar(out, out, out(func(in) = out is det)).
e_bar(2, 3, (func(X) = X * X)).
    % ok (T = int)

:- func call_e_bar = int.
call_e_bar = F(X) + F(Y) :- e_bar(X, Y, F).
    % ok
    % returns 13 (= 2*2 + 3*3)

main(!IO) :-
    call_foo,
    call_e_foo,
    io.write_line(call_bar, !IO),
    io.write_line(call_e_bar, !IO).
