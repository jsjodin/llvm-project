% MANUAL: Exception handling
% A thrown value is unified with each catch's term in turn; the first that
% unifies runs its goal, and the other catch and catch_any branches are
% ignored. A catch of another type does not match.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:then 6
% OUTPUT-NEXT:seven
% OUTPUT-NEXT:string negative
% OUTPUT-NOT:{{.}}

:- module run_catch_in_turn.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

:- pred p(int::in, int::out) is det.
p(N, M) :-
    ( if N = 7 then
        throw(7)
    else if N < 0 then
        throw("negative")
    else
        M = N * 2
    ).

:- pred try_p(int::in, io::di, io::uo) is cc_multi.
try_p(N, !IO) :-
    ( try [] p(N, X)
    then
        io.format("then %d\n", [i(X)], !IO)
    catch "positive" ->
        io.write_string("positive\n", !IO)
    catch 7 ->
        io.write_string("seven\n", !IO)
    catch I ->
        io.format("int %d\n", [i(I)], !IO)
    catch S ->
        io.write_string("string " ++ S ++ "\n", !IO)
    catch_any _ ->
        io.write_string("any\n", !IO)
    ).

main(!IO) :-
    try_p(3, !IO),
    try_p(7, !IO),
    try_p(-1, !IO).
