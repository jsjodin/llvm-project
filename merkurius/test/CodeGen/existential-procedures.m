% Existentially typed predicates and functions give their callers, before their
% other outputs, a type_info for each quantified type variable no existential
% constraint has, and a typeclass_info for each constraint; a clause has the type
% it chose where its callers have a word. The output is mmc's for the same
% program but for two things mmc does not do: it rejects sw's clauses with its
% constraint, which there are one clause with a disjunction, and it kills the
% program at listed's io.write, through a type_info for T that is list(T)'s (the
% design document's register, for both).
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module existential_procedures.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list.

:- typeclass sh(T) where [ func sz(T) = int, func nm(T) = string ].
:- typeclass big(T) <= sh(T) where [ func bg(T) = int ].
:- instance sh(int) where [ sz(X) = X, nm(_) = "int" ].
:- instance sh(float) where [ sz(X) = truncate_to_int(X), nm(_) = "float" ].
:- instance sh(list(T)) <= sh(T) where [ sz(L) = len(L), nm(_) = "list" ].
:- instance big(int) where [ bg(X) = X * 100 ].
:- type uu ---> uu.
:- instance sh(uu) where [ sz(_) = 77, nm(_) = "uu" ].
:- func len(list(T)) = int <= sh(T).
len([]) = 0.
len([H | T]) = sz(H) + len(T).

% Constrained, in a float register in the clause, a word for the caller.
:- some [T] pred mk(int::in, T::out) is det => sh(T).
mk(X, X * 2).
:- some [T] pred mkf(int::in, T::out) is det => sh(T).
mkf(X, float(X) + 0.5).
% No constraint: a type_info.
:- some [T] func any(int) = T.
any(_) = "str".
:- some [T] func fany(int) = T.
fany(X) = float(X) * 2.0.
% A type with no values.
:- some [T] pred mku(T::out) is det => sh(T).
mku(uu).
% Two variables, one constrained.
:- some [T, U] pred pair2(int::in, T::out, U::out) is det => sh(T).
pair2(N, N, "s").
% A type built from the given one; universal and existential.
:- all [A] some [T] pred wrap(A::in, list(A)::out, T::out) is det.
wrap(X, [X], {X, X}).
% Gives on a type the caller gave it, by a constraint of its own.
:- all [A] some [T] (pred via(A::in, T::out) is det => sh(T)) <= sh(A).
via(X, [X]).
% From a call: passing on what a call gave, and a list of it.
:- some [T] pred on(int::in, T::out) is det => sh(T).
on(X, Y) :- mk(X, Y).
:- some [T] pred listed(int::in, list(T)::out) is det => sh(list(T)).
listed(X, [Y, Y]) :- mk(X, Y).
% Multiple clauses, a switch.
:- type en ---> ea ; eb ; ec.
:- some [T] pred sw(en::in, T::out) is det => sh(T).
sw(ea, 10).
sw(eb, 11).
sw(ec, Y) :- Y = 12.
% Semidet, with outputs in memory.
:- some [T] pred sd(int::in, T::out, int::out, int::out, int::out, int::out) is semidet => sh(T).
sd(X, 1.5, X, X, X, X) :- X > 0.
% A branch that throws: only one binds.
:- some [T] pred br(int::in, T::out) is det => sh(T).
br(X, Y) :- ( if X > 0 then mk(X, Y) else throw_it ).
:- pragma foreign_decl("C", "#include <stdlib.h>").
:- pred throw_it is erroneous.
:- pragma foreign_proc("C", throw_it, [will_not_call_mercury, promise_pure], "abort();").
% A superclass of what was given, a method's constraint at a given type.
:- some [T] pred bigd(int::in, T::out) is det => big(T).
bigd(X, X).
:- pred usebig(T::in, io::di, io::uo) is det <= big(T).
usebig(X, !IO) :- io.write_int(bg(X), !IO), io.write_string(" ", !IO), io.write_int(sz(X), !IO), io.write_string(" ", !IO), io.write(X, !IO), io.nl(!IO).
:- pred use(T::in, io::di, io::uo) is det <= sh(T).
use(X, !IO) :- io.write_string(nm(X), !IO), io.write_string(" ", !IO), io.write_int(sz(X), !IO), io.write_string(" ", !IO), io.write(X, !IO), io.nl(!IO).
:- pred same(T::in, T::in, io::di, io::uo) is det.
same(X, Y, !IO) :- ( if X = Y then io.write_string("eq ", !IO) else io.write_string("ne ", !IO) ), compare(R, X, Y), io.write(R, !IO), io.nl(!IO).

main(!IO) :-
    mk(3, A), use(A, !IO), use([A, A], !IO),
    mkf(3, B), use(B, !IO),
    C = any(1), io.write(C, !IO), io.nl(!IO),
    D = fany(4), io.write(D, !IO), io.nl(!IO),
    mku(E), use(E, !IO),
    pair2(5, F, G), use(F, !IO), io.write(G, !IO), io.nl(!IO),
    wrap(1.5, H, I), io.write(H, !IO), io.write(I, !IO), io.nl(!IO),
    via(3, J), use(J, !IO),
    via([2.5], J2), use(J2, !IO),
    on(6, K), use(K, !IO),
    listed(7, L), use(L, !IO), io.write(L, !IO), io.nl(!IO),
    sw(ea, M0), use(M0, !IO),
    sw(eb, M1), use(M1, !IO),
    sw(ec, M2), use(M2, !IO),
    ( if sd(4, O, P, Q, R, S) then use(O, !IO), io.write_int(P + Q + R + S, !IO), io.nl(!IO) else io.write_string("no\n", !IO) ),
    ( if sd(-4, O2, _, _, _, _) then use(O2, !IO) else io.write_string("no\n", !IO) ),
    br(8, T), use(T, !IO),
    bigd(9, U), usebig(U, !IO), use(U, !IO),
    % Tests and comparisons of what was given, directly and inside a type.
    mk(3, V1), same(V1, V1, !IO),
    ( if V1 = V1 then io.write_string("eq\n", !IO) else true ),
    W = [V1, V1], same(W, [V1], !IO),
    X1 = V1, Y1 = X1,
    ( if Y1 = X1 then io.write_string("eq\n", !IO) else true ),
    ( if W = [V1] then io.write_string("one\n", !IO) else io.write_string("two\n", !IO) ),
    compare(Ord, W, [V1]), io.write(Ord, !IO), io.nl(!IO),
    ( if [Z1 | _] = W then use(Z1, !IO) else true ),
    ( if mk(1, V5), sz(V5) = 2 then use(V5, !IO) else io.write_string("not 2\n", !IO) ).

% The hidden outputs come first; a float the clause chose is a word for callers.
% MERK-LABEL: merk.proc private @existential_procedures.mkf_2_0(%arg0: i64) -> (!merk.typeclass_info, !merk.word)
% MERK: merk.to_word
% MERK-LABEL: merk.proc private @existential_procedures.any_f1_0(%arg0: i64) -> (!merk.type_info, !merk.word)

% OUTPUT: int 6 6
% OUTPUT-NEXT: list 12 [6, 6]
% OUTPUT-NEXT: float 3 3.5
% OUTPUT-NEXT: "str"
% OUTPUT-NEXT: 8.0
% OUTPUT-NEXT: uu 77 uu
% OUTPUT-NEXT: int 5 5
% OUTPUT-NEXT: "s"
% OUTPUT-NEXT: [1.5]{1.5, 1.5}
% OUTPUT-NEXT: list 3 [3]
% OUTPUT-NEXT: list 2 {{\[\[}}2.5]]
% OUTPUT-NEXT: int 12 12
% OUTPUT-NEXT: list 28 [14, 14]
% OUTPUT-NEXT: [14, 14]
% OUTPUT-NEXT: int 10 10
% OUTPUT-NEXT: int 11 11
% OUTPUT-NEXT: int 12 12
% OUTPUT-NEXT: float 1 1.5
% OUTPUT-NEXT: 16
% OUTPUT-NEXT: no
% OUTPUT-NEXT: int 16 16
% OUTPUT-NEXT: 900 9 9
% OUTPUT-NEXT: int 9 9
% OUTPUT-NEXT: eq '='
% OUTPUT-NEXT: eq
% OUTPUT-NEXT: ne '>'
% OUTPUT-NEXT: eq
% OUTPUT-NEXT: two
% OUTPUT-NEXT: '>'
% OUTPUT-NEXT: int 6 6
% OUTPUT-NEXT: int 2 2
