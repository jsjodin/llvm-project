% MANUAL: Source file name
% source_file pragmas and line number directives, which change where
% messages say the lines are (Parser/source-file.m), but not what a
% program does.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:42
% OUTPUT-NOT:{{.}}

:- module run_source_file.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.
:- pragma source_file("generator.m.in").
#20
:- func f(int) = int.
f(X) = X + 1.
:- pragma source_file("run-source-file.m").
#18
main(!IO) :- io.write_int(f(41), !IO), io.nl(!IO).
