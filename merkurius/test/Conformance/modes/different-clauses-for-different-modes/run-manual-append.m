% MANUAL: Different clauses for different modes
% Clauses whose head arguments are annotated with modes apply to those
% modes only. The manual's append, whose clauses for (out, in, in) use
% another algorithm, with promise_equivalent_clauses.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[1, 2, 3]
% OUTPUT-NEXT:[{[], [1, 2]}, {[1], [2]}, {[1, 2], []}]
% OUTPUT-NEXT:[2, 3]
% OUTPUT-NEXT:[1, 2]
% OUTPUT-NEXT:no prefix
% OUTPUT-NOT:{{.}}

:- module run_manual_append.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- pred app(list(T), list(T), list(T)).
:- mode app(in, in, out) is det.
:- mode app(out, out, in) is multi.
:- mode app(in, out, in) is semidet.
:- mode app(out, in, in) is semidet.
:- pragma promise_equivalent_clauses(app/3).

app(L1::in, L2::in, L3::out) :- usual_app(L1, L2, L3).
app(L1::out, L2::out, L3::in) :- usual_app(L1, L2, L3).
app(L1::in, L2::out, L3::in) :- usual_app(L1, L2, L3).
app(L1::out, L2::in, L3::in) :- other_app(L1, L2, L3).

:- pred usual_app(list(T), list(T), list(T)).
:- mode usual_app(in, in, out) is det.
:- mode usual_app(out, out, in) is multi.
:- mode usual_app(in, out, in) is semidet.
usual_app([], Ys, Ys).
usual_app([X | Xs], Ys, [X | Zs]) :-
    usual_app(Xs, Ys, Zs).

:- pred other_app(list(T)::out, list(T)::in, list(T)::in) is semidet.
other_app(Prefix, Suffix, List) :-
    list.length(List, ListLength),
    list.length(Suffix, SuffixLength),
    PrefixLength = ListLength - SuffixLength,
    list.split_list(PrefixLength, List, Prefix, Suffix).

main(!IO) :-
    app([1, 2], [3], L),
    io.write_line(L, !IO),
    solutions((pred({A, B}::out) is multi :- app(A, B, [1, 2])), Splits),
    io.write_line(Splits, !IO),
    ( if app([1], S, [1, 2, 3]) then
        io.write_line(S, !IO)
    else
        io.write_string("no suffix\n", !IO)
    ),
    ( if app(P, [3], [1, 2, 3]) then
        io.write_line(P, !IO)
    else
        io.write_string("no prefix\n", !IO)
    ),
    ( if app(Q, [4], [1, 2, 3]) then
        io.write_line(Q, !IO)
    else
        io.write_string("no prefix\n", !IO)
    ).
