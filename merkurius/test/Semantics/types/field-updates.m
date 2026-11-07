% A field update may change the type of a type parameter that occurs in the
% field's type and in no other field of its constructor, whatever the
% type's other constructors have, as mmc lets it (the register,
% "Type-changing field updates"); not one that another field shares, nor
% a phantom parameter, which no field has, nor another parameter.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module field_updates.
:- interface.
:- import_module list.

:- type box(T) ---> box(contents :: T, count :: int).
:- type two(T) ---> two(a :: T, b :: T).
:- type u(T) ---> one(x :: T) ; other(y :: int, z :: T).
:- type ph(T) ---> ph(v :: int).
:- type nest(T) ---> nest(l :: list(T), n :: int).
:- type kv(K, V) ---> kv(k :: K, val :: V).

:- pred changes(box(string)::in, box(int)::out) is det.
:- pred shared(two(string)::in, two(int)::out) is det.
:- pred other_constructor(u(string)::in, u(int)::out) is semidet.
:- pred phantom(ph(string)::in, ph(int)::out) is det.
:- pred nested(nest(string)::in, nest(int)::out) is det.
:- pred one_of_two(kv(int, string)::in, kv(int, float)::out) is det.
:- pred not_the_other(kv(int, string)::in, kv(string, float)::out) is det.
:- func polymorphic(box(T), U) = box(U).

:- implementation.

changes(X, X ^ contents := 5).
shared(X, X ^ a := 1).
% CHECK-DAG: field-updates.m:[[@LINE-1]]:11: error: type error: `X ^ a := 1' must have type field_updates.two(int) here, which no declaration of a :=/2 fits
other_constructor(X, X ^ x := 1).
phantom(X, X ^ v := 1).
% CHECK-DAG: field-updates.m:[[@LINE-1]]:12: error: type error: `X ^ v := 1' must have type field_updates.ph(int) here, which no declaration of v :=/2 fits
nested(X, X ^ l := [1]).
one_of_two(X, X ^ val := 1.5).
not_the_other(X, X ^ val := 1.5).
% CHECK-DAG: field-updates.m:[[@LINE-1]]:18: error: type error: `X ^ val := 1.5' must have type field_updates.kv(string, float) here, which no declaration of val :=/2 fits
polymorphic(B, X) = B ^ contents := X.
