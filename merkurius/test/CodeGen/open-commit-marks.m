% A commit releases the trail marks that its goal's choice points left,
% even when the procedure that commits has no open variable of its own and
% the marks are a callee's: here gen's disjunction over a local open term,
% left at its first solution by first's promise_equivalent_solutions. Each
% call of first left one live mark before, so that every later binding was
% trailed and a collection would refuse the trail. The live marks are read
% from the runtime.
%
% RUN: merkc -fopen-types %s -o %t && %t | FileCheck %s --match-full-lines
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && %t.O0 | FileCheck %s --match-full-lines
% CHECK:5
% CHECK-NEXT:live marks 0

:- module open_commit_marks.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, string.

:- pragma foreign_decl("C", "#include <stdint.h>
uint64_t merk_open_trail_marks(void);").
:- pred live_marks(int::out) is det.
:- pragma foreign_proc("C", live_marks(N::out),
    [will_not_call_merkurius, promise_pure],
    "N = (MR_Integer)merk_open_trail_marks();").

:- pred pick(open(list(int))::ia, int::out) is nondet.
pick(L, X) :-
    ( L = [X | _]
    ; L = [_, X | _]
    ).

:- pred gen(int::out) is nondet.
gen(X) :-
    L = to_open([1, 2, 3]),
    pick(L, X).

:- pred first(int::out) is semidet.
first(X) :- promise_equivalent_solutions [X] gen(X).

:- pred count(int::in, int::in, int::out) is det.
count(N, A0, A) :-
    ( if N = 0 then
        A = A0
    else
        ( if first(X) then A1 = A0 + X else A1 = A0 ),
        count(N - 1, A1, A)
    ).

main(!IO) :-
    count(5, 0, A),
    io.write_int(A, !IO), io.nl(!IO),
    live_marks(M),
    io.format("live marks %d\n", [i(M)], !IO).
