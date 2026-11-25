% A hole made by the inline allocation is the variable cell's tagged
% reference, and statepoint lowering keeps the untagged cell as its base,
% an update-only root. Once the hole is bound to another, a typed root that
% reaches it replaces the reference by the binding, which skipped the
% cell: the base then pointed at a cell nothing had forwarded, and the
% collector stopped with "an update-only root whose cell nothing else
% reaches". A cell that an update-only root points at is now copied with
% its binding, as a trailed one is. The output is checked by hand: the
% three holes are one variable, bound to 7.
%
% RUN: merkc -fopen-types %s -o %t && env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fopen-types -O0 %s -o %t.O0 && env MERK_GC_EVERY_ALLOC=1 %t.O0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module open_gc_bound_base.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open.

:- pred show(open(T)::ia, io::di, io::uo) is det.
show(X, !IO) :-
    promise_pure ( impure open.write(X, !IO), io.nl(!IO) ).

    % The holes are passed unbound, so the unifications bind one variable
    % to another at run time.
:- pred same(open(int)::ia, open(int)::ia) is semidet.
same(X, X).

:- pred bind(open(int)::ia, int::in) is semidet.
bind(X, N) :-
    X = to_open(N).

    % A is bound to B before the list's cells are allocated, each a
    % collection with MERK_GC_EVERY_ALLOC.
:- pred chain(open(list(int))::oa) is semidet.
chain(L) :-
    L = [A, B, C],
    same(A, B),
    same(C, B),
    bind(C, 7).

main(!IO) :-
    ( if chain(L) then show(L, !IO) else io.write_string("no\n", !IO) ).
% OUTPUT:[7, 7, 7]
% OUTPUT-NOT:{{.}}
