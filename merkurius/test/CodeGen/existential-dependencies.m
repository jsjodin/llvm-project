% An existentially quantified type variable in no argument's type, which a
% constraint determines through a functional dependency (reference manual,
% "Existentially typed data types"): its type_info is the one in the
% typeclass_info, for a call of an existentially typed predicate and for a
% deconstruction. The type checker gives polymorphism analysis the type the
% goal gave it. The output is mmc's for the same program.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -O2 %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module existential_dependencies.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- typeclass xable(T1, T2) <= (T1 -> T2) where [ func x(T1) = T2 ].
:- instance xable(int, string) where [
    x(N) = ( if N = 1 then "one" else "other" )
].

:- some [T1, T2] pred mk(T1::out) is det => xable(T1, T2).
mk(1).

:- type t3 ---> some [T1, T2] f5(T1) => xable(T1, T2).

:- func wrap = t3.
wrap = 'new f5'(2).

main(!IO) :-
    mk(A),
    io.write_line(x(A), !IO),
    W = wrap,
    W = f5(B),
    io.write_line(x(B), !IO),
    io.write_line(B, !IO),
    io.write_line(W, !IO).

% OUTPUT: "one"
% OUTPUT-NEXT: "other"
% OUTPUT-NEXT: 2
% OUTPUT-NEXT: f5(2)
% OUTPUT-NOT: {{.}}
