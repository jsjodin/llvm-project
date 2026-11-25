% MANUAL: Trace goals
% Run-time conditions, of an environment variable that is set and of one
% that is not, with and, or and not, and with a compile-time condition.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:PATH set
% OUTPUT-NEXT:not unset
% OUTPUT-NEXT:or
% OUTPUT-NEXT:end
% OUTPUT-NOT:{{.}}

:- module run_run_time.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :-
    trace [run_time(env("PATH")), io(!S)] ( io.write_string("PATH set\n", !S) ),
    trace [run_time(env("MERK_SURELY_UNSET_VAR")), io(!S)] ( io.write_string("unset set\n", !S) ),
    trace [run_time(not env("MERK_SURELY_UNSET_VAR")), io(!S)] ( io.write_string("not unset\n", !S) ),
    trace [run_time(env("PATH") and env("MERK_SURELY_UNSET_VAR")), io(!S)] ( io.write_string("and\n", !S) ),
    trace [run_time(env("PATH") or env("MERK_SURELY_UNSET_VAR")), io(!S)] ( io.write_string("or\n", !S) ),
    trace [compile_time(flag("a")), run_time(env("PATH")), io(!S)] ( io.write_string("both\n", !S) ),
    io.write_string("end\n", !IO).
