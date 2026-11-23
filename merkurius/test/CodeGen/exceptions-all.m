% try_all and finally (docs/INPROGRESS.md, Exceptions, step 2). A try_all
% of a multi or nondet goal calls the runtime's merk_try_nondet, which keeps
% its handler while each solution goes on to collect_all's step, so the
% solutions before an exception stay in the list; a det or semidet goal's
% is a try. Many solutions, each a new cell, run with a collection at every
% allocation, whose walk goes through merk_try_nondet's frame while the
% step runs; a try_all nests in another's goal. finally runs the cleanup,
% then throws the goal's exception again, or the cleanup's. OUTPUT is mmc's.
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module exceptions_all.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, maybe, string, univ.

:- pred d(int::out) is det.
d(1).
:- pred dt(int::out) is det.
dt(X) :- ( if 1 > 0 then throw("dt") else X = 0 ).
:- pred s(int::in, int::out) is semidet.
s(X, Y) :- X > 0, ( if X > 5 then throw("s") else Y = X ).
:- pred m(int::out) is multi.
m(1). m(2). m(3).
:- pred n(int::in, int::out) is nondet.
n(T, Y) :- list.member(Y, [1, 2, 3, 4]), ( if Y = T then throw(Y) else Y \= 2 ).

:- pred show(maybe(univ)::in, list(int)::in, io::di, io::uo) is det.
show(M, L, !IO) :-
    ( M = no, io.write_string("no ", !IO)
    ; M = yes(U), io.write_string("yes(", !IO), io.write(univ_value(U), !IO), io.write_string(") ", !IO) ),
    io.write(L, !IO), io.nl(!IO).

:- pred p(int::out, io::di, io::uo) is det.
p(X, !IO) :- io.write_string("p\n", !IO), X = 5.
:- pred pt(int::out, io::di, io::uo) is det.
pt(X, !IO) :- io.write_string("pt\n", !IO), ( if 1 > 0 then throw("pt") else X = 0 ).
:- pred c(io.res::out, io::di, io::uo) is det.
c(ok, !IO) :- io.write_string("cleanup\n", !IO).
:- pred ct(io.res::out, io::di, io::uo) is det.
ct(R, !IO) :- io.write_string("cleanup throws\n", !IO), ( if 1 > 0 then throw("ct") else R = ok ).

:- pred fin(int::out, io::di, io::uo) is det.
fin(X, !IO) :- finally(pt, X, c, _, !IO).
:- pred fin2(int::out, io::di, io::uo) is det.
fin2(X, !IO) :- finally(pt, X, ct, _, !IO).

    % Many solutions, each a fresh cell, then an exception.
:- pred many(int::in, list(int)::out) is nondet.
many(N, [X, X * 2]) :-
    list.member(X, 1 .. N),
    ( if X = N then Msg = "at " ++ string.int_to_string(X), throw(Msg) else true ).

    % A try_all inside a try_all's goal, with an inner exception caught and
    % an outer one after it.
:- pred nested(int::out) is nondet.
nested(S) :-
    list.member(K, [1, 2, 3]),
    promise_equivalent_solutions [M, L] (
        try_all(many(K * 100), M, L)
    ),
    ( if K = 3 then Len = list.length(L) : int, throw(Len) else true ),
    M = yes(_),
    S = list.foldl(func(P, A) = A + list.det_head(P), L, 0).

:- pred show_maybe(maybe(univ)::in, io::di, io::uo) is det.
show_maybe(no, !IO) :- io.write_string("no", !IO).
show_maybe(yes(U), !IO) :- io.write(univ_value(U), !IO).

main(!IO) :-
    try_all(d, M1, L1), show(M1, L1, !IO),
    try_all(dt, M2, L2), show(M2, L2, !IO),
    try_all(s(0), M3, L3), show(M3, L3, !IO),
    try_all(s(3), M4, L4), show(M4, L4, !IO),
    try_all(s(9), M5, L5), show(M5, L5, !IO),
    try_all(m, M6, L6), show(M6, L6, !IO),
    try_all(n(9), M7, L7), show(M7, L7, !IO),
    try_all(n(3), M8, L8), show(M8, L8, !IO),
    try_all(n(1), M9, L9), show(M9, L9, !IO),
    finally(p, X1, c, R1, !IO), io.write_int(X1, !IO), io.write(R1, !IO), io.nl(!IO),
    try_io(fin, T1, !IO), ( T1 = exception(U1) -> io.write(univ_value(U1), !IO) ; io.write_string("ok", !IO) ), io.nl(!IO),
    try_io(fin2, T2, !IO), ( T2 = exception(U2) -> io.write(univ_value(U2), !IO) ; io.write_string("ok", !IO) ), io.nl(!IO),
    try_all(many(20000), M10, L10),
    show_maybe(M10, !IO),
    io.format(" %d %d\n", [i(length(L10)),
        i(list.foldl(func(P, A) = A + list.det_last(P), L10, 0))], !IO),
    try_all(nested, M11, L11),
    show_maybe(M11, !IO), io.write_string(" ", !IO), io.write(L11, !IO), io.nl(!IO).
% OUTPUT:no [1]
% OUTPUT-NEXT:yes("dt") []
% OUTPUT-NEXT:no []
% OUTPUT-NEXT:no [3]
% OUTPUT-NEXT:yes("s") []
% OUTPUT-NEXT:no [1, 2, 3]
% OUTPUT-NEXT:no [1, 3, 4]
% OUTPUT-NEXT:yes(3) [1]
% OUTPUT-NEXT:yes(1) []
% OUTPUT-NEXT:p
% OUTPUT-NEXT:cleanup
% OUTPUT-NEXT:5ok
% OUTPUT-NEXT:pt
% OUTPUT-NEXT:cleanup
% OUTPUT-NEXT:"pt"
% OUTPUT-NEXT:pt
% OUTPUT-NEXT:cleanup throws
% OUTPUT-NEXT:"ct"
% OUTPUT-NEXT:"at 20000" 19999 399980000
% OUTPUT-NEXT:299 [4950, 19900]
