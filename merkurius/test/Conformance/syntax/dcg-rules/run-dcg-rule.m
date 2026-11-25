% MANUAL: DCG-rules
% A DCG rule with an argument of its own, and its two implicit arguments.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{2, ['b']}
% OUTPUT-NOT:{{.}}

:- module run_dcg_rule.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, int, list, string.

:- pred count(int::in, int::out, list(char)::in, list(char)::out) is det.
count(N0, N) --> ( ['a'] -> count(N0 + 1, N) ; { N = N0 } ).

main(!IO) :-
    count(0, N, ['a', 'a', 'b'], Rest),
    io.write({N, Rest}, !IO), io.nl(!IO).
