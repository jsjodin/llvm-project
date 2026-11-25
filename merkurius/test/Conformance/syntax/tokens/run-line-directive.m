% MANUAL: Tokens
% A line number directive sets the number of the next line, which
% $line gives.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:100
% OUTPUT-NEXT:101
% OUTPUT-NOT:{{.}}

:- module run_line_directive.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
#100
main(!IO) :- io.write_int($line, !IO),
    io.nl(!IO), io.write_int($line, !IO), io.nl(!IO).
