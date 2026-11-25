% MANUAL: Character set
% UTF-8 in strings, quoted names, character literals and comments
% (åäö, λ, ∃).
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:åäö λ ∃ 😀
% OUTPUT-NEXT:9
% OUTPUT-NEXT:{'ångström', 'λ', 'é'}
% OUTPUT-NOT:{{.}}

:- module run_utf8.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type t ---> 'ångström' ; 'λ'.

main(!IO) :-
    S = "åäö λ ∃ 😀",
    io.write_string(S, !IO), io.nl(!IO),
    io.write_int(string.count_codepoints(S), !IO), io.nl(!IO),
    io.write({'ångström' : t, 'λ' : t, 'é'}, !IO), io.nl(!IO).
