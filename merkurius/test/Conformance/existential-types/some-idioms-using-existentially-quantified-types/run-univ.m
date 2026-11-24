% MANUAL: Some idioms using existentially quantified types
% The manual's idioms: univ and univ_value give an existentially typed
% predicate values of different types at each call, and a type of its own
% like univ does so with a type class constraint, as do mode-specific
% clauses through an intermediate predicate.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:42
% OUTPUT-NEXT:"blah"
% OUTPUT-NEXT:"42"
% OUTPUT-NEXT:42
% OUTPUT-NOT:{{.}}

:- module run_univ.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module string, univ.

:- typeclass showable(T) where [ func show(T) = string ].
:- instance showable(int) where [ show(I) = string.int_to_string(I) ].
:- instance showable(string) where [ show(S) = S ].

:- some [T] pred good_example(string, T).
:- mode good_example(in, out) is semidet.

good_example(Name, univ_value(Univ)) :-
    ( Name = "foo", Univ = univ(42)
    ; Name = "bar", Univ = univ("blah")
    ).

:- type univ_showable
    --->    some [T] (mkshowable(T) => showable(T)).

:- some [T] pred harder_example(string, T) => showable(T).
:- mode harder_example(in, out) is semidet.

harder_example(Name, Showable) :-
    ( Name = "bar", Univ = 'new mkshowable'(42)
    ; Name = "foo", Univ = 'new mkshowable'("blah")
    ),
    Univ = mkshowable(Showable).

:- some [T] pred good_example3(string, T).
:-          mode good_example3(in(bound("foo")), out) is det.
:-          mode good_example3(in(bound("bar")), out) is det.
good_example3(Name, univ_value(Univ)) :-
    good_example3_univ(Name, Univ).

:- pred good_example3_univ(string, univ).
:- mode good_example3_univ(in(bound("foo")), out) is det.
:- mode good_example3_univ(in(bound("bar")), out) is det.
:- pragma promise_pure(good_example3_univ/2).
good_example3_univ("foo"::in(bound("foo")), univ(42)::out).
good_example3_univ("bar"::in(bound("bar")), univ("blah")::out).

main(!IO) :-
    ( if good_example("foo", A) then io.write_line(A, !IO) else true ),
    ( if good_example("bar", B) then io.write_line(B, !IO) else true ),
    ( if harder_example("bar", C) then
        io.write_line(show(C), !IO)
    else
        true
    ),
    good_example3("foo", D),
    io.write_line(D, !IO).
