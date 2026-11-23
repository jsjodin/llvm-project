% The standard library module univ. The declarations are those of the
% library reference manual (22.01), less those that need modules not stubbed
% yet; see README.md. A univ is an existentially typed cell, which holds its
% value's type_info, so a conversion back compares type_infos.

:- module univ.
:- interface.

:- type univ.
:- pred type_to_univ(T, univ).
:- mode type_to_univ(di, uo) is det.
:- mode type_to_univ(in, out) is det.
:- mode type_to_univ(out, in) is semidet.
:- pred univ_to_type(univ, T).
:- mode univ_to_type(in, out) is semidet.
:- mode univ_to_type(out, in) is det.
:- mode univ_to_type(uo, di) is det.
:- func univ(T) = univ.
:- mode univ(in) = out is det.
:- mode univ(di) = uo is det.
:- mode univ(out) = in is semidet.
:- pred det_univ_to_type(univ::in, T::out) is det.
:- some [T] func univ_value(univ) = T.

:- implementation.
:- import_module exception, string.

:- type univ
    --->    some [T] univ_cons(T).

:- pragma promise_equivalent_clauses(type_to_univ/2).

type_to_univ(T::di, Univ::uo) :-
    Univ = 'new univ_cons'(T).
type_to_univ(T::in, Univ::out) :-
    Univ = 'new univ_cons'(T).
type_to_univ(T::out, Univ::in) :-
    Univ = univ_cons(Value),
    same_type_value(Value, T).

univ_to_type(Univ, T) :-
    type_to_univ(T, Univ).

univ(T) = Univ :-
    type_to_univ(T, Univ).

det_univ_to_type(Univ, T) :-
    ( if univ_to_type(Univ, T0) then
        T = T0
    else
        Univ = univ_cons(Value),
        throw(software_error("det_univ_to_type: conversion failed\n" ++
            "\tUniv Type: " ++ value_type_name(Value) ++ "\n" ++
            "\tObject Type: " ++ type_name_of(T)))
    ).

univ_value(univ_cons(T)) = T.

    % The name of T, as type_desc.type_name gives it.
:- func type_name_of(T::unused) = (string::out) is det.
:- pragma foreign_proc("C",
    type_name_of(_Value::unused) = (Name::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Name = merk_type_name(TypeInfo_for_T);
").

    % The name of a value's type.
:- func value_type_name(T::in) = (string::out) is det.
:- pragma foreign_proc("C",
    value_type_name(_Value::in) = (Name::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    Name = merk_type_name(TypeInfo_for_T);
").

    % The value as a U, if U is its type T.
:- pred same_type_value(T::in, U::out) is semidet.
:- pragma foreign_proc("C",
    same_type_value(Value::in, Same::out),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR =
        merk_compare_type_infos(TypeInfo_for_T, TypeInfo_for_U) == 0;
    Same = Value;
").

:- pragma foreign_decl("C", local, "
int merk_compare_type_infos(MR_Word x, MR_Word y);
char *merk_type_name(MR_Word type_info);
").
