% Unique modes that mode analysis accepts, the first of their two phases
% (Overton, "Precise and Expressive Mode Systems for Typed Logic
% Programming Languages", 2003, sections 3.2.1 and 3.2.3; reference
% manual, "Unique modes"): a unique value passed on, as the I/O state is;
% a deconstruction, which keeps the term's uniqueness, as mmc does; a
% part that the term and its variable both use after, which is shared,
% and one that only one of them uses, which stays unique; an assignment
% of a value used after, which shares it; a construction, whose argument
% keeps its inst, while the term's copy of it is shared if the argument
% is used after; and a value that a call clobbers, unused after. These
% are mmc's rules (-fmmc-uniqueness): by default, an argument used after
% is shared too if the term is (../uniqueness/aliased.m).
%
% RUN: merkc -fc1 -fdebug-dump-modes -fmmc-uniqueness %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module uniqueness.
:- interface.
:- import_module io, list.

:- pred main(io::di, io::uo) is det.
:- pred keep(list(int)::di, list(int)::uo) is semidet.
:- pred parts(list(int)::di, list(int)::uo) is semidet.
:- pred shared_part(list(list(int))::di, list(int)::out,
    list(list(int))::out) is semidet.
:- pred assigned(list(int)::di, list(int)::out, list(int)::out) is det.
:- pred built(list(int)::uo, list(list(int))::out) is det.
:- pred clobbered is det.

:- implementation.

:- pred d(list(int)::di) is det.
d(_).

main(!IO) :-
    io.write_int(1, !IO),
    io.nl(!IO).
% CHECK-LABEL: pred uniqueness.main/2 mode 0: main(di, uo) is det
% CHECK: call pred io.write_int/3 mode 0: write_int(V_1, IO!0, IO!1)
% CHECK: call pred io.nl/2 mode 0: nl(IO!1, IO!2)
% CHECK: final: IO!0 clobbered, IO!2 unique

keep(X, Y) :-
    X = [_ | _],
    Y = X.
% CHECK-LABEL: pred uniqueness.keep/2 mode 0: keep(di, uo) is semidet
% CHECK: final: X unique('[|]'(unique, unique)), Y unique('[|]'(unique, unique))

parts(X, Y) :-
    X = [H | T],
    Y = [H | T].
% CHECK-LABEL: pred uniqueness.parts/2 mode 0: parts(di, uo) is semidet
% CHECK: final: X unique('[|]'(unique, unique)), Y unique('[|]'(unique, unique))

shared_part(L, X, L1) :-
    L = [X | _],
    L1 = L.
% CHECK-LABEL: pred uniqueness.shared_part/3 mode 0: shared_part(di, out, out) is semidet
% CHECK: final: L unique('[|]'(ground, unique)), X ground, L1 unique('[|]'(ground, unique))

assigned(X, Y, Z) :-
    Y = X,
    Z = X.
% CHECK-LABEL: pred uniqueness.assigned/3 mode 0: assigned(di, out, out) is det
% CHECK: final: X ground, Y ground, Z ground

built(X, L) :-
    X = [1],
    L = [X].
% CHECK-LABEL: pred uniqueness.built/2 mode 0: built(uo, out) is det
% CHECK: final: X unique('[|]'(unique(1), unique([]))), L unique('[|]'(bound('[|]'(bound(1), bound([]))), unique([])))

clobbered :-
    X = [1],
    d(X).
% CHECK-LABEL: pred uniqueness.clobbered/0 mode 0: clobbered is det
% CHECK: call pred uniqueness.d/1 mode 0: d(X)
