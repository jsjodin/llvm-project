% MANUAL: Tokens
% Quoted names with a doubled quote and escapes, and an operator as a
% character literal in parentheses.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{'it\'s', 'tab\there', plain, 'two words', ':'}
% OUTPUT-NOT:{{.}}

:- module run_names.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type t ---> 'it''s' ; 'tab\there' ; 'plain' ; 'two words'.

main(!IO) :-
    C = (':'),
    io.write({'it''s', 'tab\there', 'plain', 'two words', C}, !IO), io.nl(!IO).
