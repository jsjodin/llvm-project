% Equality of closed(T) values inside open terms (design document, "Open
% terms", Operations): the runtime's walker compares a closed field's ground
% value by its type, a float's or a string's by value, a notag's by its
% argument, and a record's field by field, the last without recursion.
%
% RUN: merkc -fopen-types -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fopen-types -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module open_closed_equal.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, float, int, list, open, string.

:- type d ---> d.
:- type n ---> n(int).
:- type g ---> g(float, string, d, n, list(int)).
:- type h ---> h(closed(g), int).
:- type e ---> e(n).
:- type k ---> k(closed(d), closed(e)).

:- pred t(string::in, bool::in, io::di, io::uo) is det.
t(S, B, !IO) :-
    io.write_string(S, !IO),
    io.write_string(if B = yes then ": yes\n" else ": no\n", !IO).

:- pred eq(open(T)::ia, open(T)::ia) is semidet.
:- pragma no_inline(eq/2).
eq(X, Y) :- X = Y.

:- func mk(float, string, int, list(int)) = open(h).
mk(F, S, I, L) = h(closed(g(F, S, d, n(I), L)), 0).

main(!IO) :-
    A = mk(1.5, "s", 3, [1, 2]),
    t("same", (if eq(A, mk(1.5, "s", 3, [1, 2])) then yes else no), !IO),
    t("float", (if eq(A, mk(2.5, "s", 3, [1, 2])) then yes else no), !IO),
    t("string", (if eq(A, mk(1.5, "t", 3, [1, 2])) then yes else no), !IO),
    t("notag", (if eq(A, mk(1.5, "s", 4, [1, 2])) then yes else no), !IO),
    t("last", (if eq(A, mk(1.5, "s", 3, [1, 3])) then yes else no), !IO),
    t("closed d e", (if eq(k(closed(d), closed(e(n(1)))) : open(k), k(closed(d), closed(e(n(1))))) then yes else no), !IO),
    t("closed e ne", (if eq(k(closed(d), closed(e(n(1)))) : open(k), k(closed(d), closed(e(n(2))))) then yes else no), !IO).
% OUTPUT:same: yes
% OUTPUT-NEXT:float: no
% OUTPUT-NEXT:string: no
% OUTPUT-NEXT:notag: no
% OUTPUT-NEXT:last: no
% OUTPUT-NEXT:closed d e: yes
% OUTPUT-NEXT:closed e ne: no
