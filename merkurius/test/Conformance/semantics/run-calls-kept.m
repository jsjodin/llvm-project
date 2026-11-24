% MANUAL: Semantics
% The default semantics is fully strict: a call that cannot succeed, or
% whose outputs nothing uses, still runs, as does a det call with no outputs.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:"erroneous call"
% OUTPUT-NEXT:"call without outputs"
% OUTPUT-NEXT:"unused output"
% OUTPUT-NOT:{{.}}

:- module run_calls_kept.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string, univ.

% Throws S, but is det: no pass may treat a call of it as erroneous.
:- func th(string, int) = int.
th(S, N) = ( if N > -1000 then throw(S) else N ).

:- pred show(exception_result(int)::in, io::di, io::uo) is det.
show(succeeded(N), !IO) :- io.format("succeeded %d\n", [i(N)], !IO).
show(failed, !IO) :- io.write_string("failed\n", !IO).
show(exception(E), !IO) :- io.write(univ_value(E), !IO), io.nl(!IO).

:- pred err(int::out) is erroneous.
err(_) :- throw("erroneous call").

:- pred no_outputs(int::in) is det.
no_outputs(N) :- ( if N > 0 then throw("call without outputs") else true ).

:- pred det_unused(int::in, int::out) is det.
det_unused(N, th("unused output", N)).

:- pred erroneous_kept(int::out) is det.
erroneous_kept(R) :- err(_), R = 1.

:- pred no_outputs_kept(int::out) is det.
no_outputs_kept(R) :- no_outputs(1), R = 1.

:- pred unused_kept(int::out) is det.
unused_kept(R) :- det_unused(1, _), R = 1.

main(!IO) :-
    try(erroneous_kept, R1), show(R1, !IO),
    try(no_outputs_kept, R2), show(R2, !IO),
    try(unused_kept, R3), show(R3, !IO).
