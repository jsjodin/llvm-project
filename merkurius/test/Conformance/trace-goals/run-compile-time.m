% MANUAL: Trace goals
% Compile-time conditions without trace flags: flags, the grade tests that
% hold for a C high-level-code compile, trace levels, and and, or and not.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:not flag b
% OUTPUT-NEXT:grade c
% OUTPUT-NEXT:grade mlds
% OUTPUT-NEXT:not deep
% OUTPUT-NEXT:end
% OUTPUT-NOT:{{.}}

:- module run_compile_time.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :-
    trace [compile_time(flag("a")), io(!S)] ( io.write_string("flag a\n", !S) ),
    trace [compile_time(flag("b")), io(!S)] ( io.write_string("flag b\n", !S) ),
    trace [compile_time(not flag("b")), io(!S)] ( io.write_string("not flag b\n", !S) ),
    trace [compile_time(flag("a") and flag("b")), io(!S)] ( io.write_string("a and b\n", !S) ),
    trace [compile_time(flag("a") or flag("b")), io(!S)] ( io.write_string("a or b\n", !S) ),
    trace [compile_time(grade(c)), io(!S)] ( io.write_string("grade c\n", !S) ),
    trace [compile_time(grade(java)), io(!S)] ( io.write_string("grade java\n", !S) ),
    trace [compile_time(grade(debug)), io(!S)] ( io.write_string("grade debug\n", !S) ),
    trace [compile_time(grade(mlds)), io(!S)] ( io.write_string("grade mlds\n", !S) ),
    trace [compile_time(grade(llds)), io(!S)] ( io.write_string("grade llds\n", !S) ),
    trace [compile_time(grade(par)), io(!S)] ( io.write_string("grade par\n", !S) ),
    trace [compile_time(tracelevel(shallow)), io(!S)] ( io.write_string("shallow\n", !S) ),
    trace [compile_time(not tracelevel(deep)), io(!S)] ( io.write_string("not deep\n", !S) ),
    io.write_string("end\n", !IO).
