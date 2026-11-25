% MANUAL: No determinism warnings
% A no_determinism_warning pragma, which keeps the warning that a
% determinism declaration could be tighter from being given.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:2
% OUTPUT-NOT:{{.}}

:- module run_no_determinism_warning.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pragma no_determinism_warning(quiet/2).
:- pred quiet(int::in, int::out) is semidet.
quiet(X, X + 1).

main(!IO) :-
    ( if quiet(1, A) then io.write_int(A, !IO) else true ),
    io.nl(!IO).
