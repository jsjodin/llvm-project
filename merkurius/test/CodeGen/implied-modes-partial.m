% Implied modes with a partially instantiated term: a term with free parts on
% the left of a call, which only tests the term's bound parts against the result.
% `e(_, yes, _) = g(K)` is built as `e(free, yes, free)` and tested after the call.
% The output is mmc's, except for nested, head, tail_element and from_pred, which
% are mode errors in mmc's own __Unify__ (design document, "A partially
% instantiated term unified with a bound one"; GitHub issue 74).
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module implied_modes_partial.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, int, list, maybe, string.

:- type e ---> e(int, bool, int).
:- type t ---> a(int, string) ; b(e, int) ; c.

:- func g(int) = e.
g(N) = e(N, ( if N > 2 then yes else no ), N * 2).

:- func h(int) = t.
h(N) = ( if N = 0 then c else if N = 1 then a(7, "x") else b(g(N), N) ).

:- func l(int) = list(int).
l(N) = [N, N + 1, N + 2].

:- func m(int) = maybe(int).
m(N) = ( if N > 0 then yes(N) else no ).

:- pred p(int::in, t::out) is det.
p(N, h(N)).

:- pred show(string::in, bool::in, io::di, io::uo) is det.
show(What, Holds, !IO) :-
    io.write_string(What ++ ": ", !IO),
    io.write(Holds, !IO),
    io.nl(!IO).

:- pred field(int::in) is semidet.
field(K) :- e(_, yes, _) = g(K).

:- pred number(int::in) is semidet.
number(K) :- e(3, _, _) = g(K).

:- pred only_functor(int::in) is semidet.
only_functor(K) :- a(_, _) = h(K).

:- pred string_field(int::in) is semidet.
string_field(K) :- a(_, "x") = h(K).

:- pred nested(int::in) is semidet.
nested(K) :- b(e(_, yes, _), _) = h(K).

:- pred head(int::in) is semidet.
head(K) :- [1, _ | _] = l(K).

:- pred tail_element(int::in) is semidet.
tail_element(K) :- [_, 3 | _] = l(K).

:- pred from_pred(int::in) is semidet.
from_pred(K) :- p(K, b(_, 2)).

:- pred any_e(int::in) is semidet.
any_e(K) :- e(_, _, _) = g(K).

:- pred bound_then_used(int::in, int::out) is semidet.
bound_then_used(K, X) :- e(X, yes, _) = g(K).

:- func holds(pred(int), int) = bool.
:- mode holds(pred(in) is semidet, in) = out is det.
holds(P, K) = ( if P(K) then yes else no ).

main(!IO) :-
    show("field 3", holds(field, 3), !IO),
    show("field 1", holds(field, 1), !IO),
    show("number 3", holds(number, 3), !IO),
    show("number 4", holds(number, 4), !IO),
    show("only_functor 1", holds(only_functor, 1), !IO),
    show("only_functor 0", holds(only_functor, 0), !IO),
    show("only_functor 5", holds(only_functor, 5), !IO),
    show("string_field 1", holds(string_field, 1), !IO),
    show("nested 5", holds(nested, 5), !IO),
    show("nested 2", holds(nested, 2), !IO),
    show("nested 1", holds(nested, 1), !IO),
    show("head 1", holds(head, 1), !IO),
    show("head 2", holds(head, 2), !IO),
    show("tail_element 2", holds(tail_element, 2), !IO),
    show("tail_element 3", holds(tail_element, 3), !IO),
    show("from_pred 2", holds(from_pred, 2), !IO),
    show("from_pred 3", holds(from_pred, 3), !IO),
    show("any_e 9", holds(any_e, 9), !IO),
    ( if bound_then_used(4, X) then
        io.write_int(X, !IO), io.nl(!IO)
    else
        io.write_string("none\n", !IO)
    ).

% CHECK:field 3: yes
% CHECK-NEXT:field 1: no
% CHECK-NEXT:number 3: yes
% CHECK-NEXT:number 4: no
% CHECK-NEXT:only_functor 1: yes
% CHECK-NEXT:only_functor 0: no
% CHECK-NEXT:only_functor 5: no
% CHECK-NEXT:string_field 1: yes
% CHECK-NEXT:nested 5: yes
% CHECK-NEXT:nested 2: no
% CHECK-NEXT:nested 1: no
% CHECK-NEXT:head 1: yes
% CHECK-NEXT:head 2: no
% CHECK-NEXT:tail_element 2: yes
% CHECK-NEXT:tail_element 3: no
% CHECK-NEXT:from_pred 2: yes
% CHECK-NEXT:from_pred 3: no
% CHECK-NEXT:any_e 9: yes
% CHECK-NEXT:4
% CHECK-NOT:{{.}}
