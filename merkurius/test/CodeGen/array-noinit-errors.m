% array.unsafe_noinit only makes arrays whose elements hold no pointers,
% so that the collector and a message's copy never read garbage as one: an
% array of strings, of a type variable's elements and of records that hold
% a string are errors at compile time. A negative size throws,
% as array.init's does.
%
% RUN: not merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s 2>&1 | FileCheck %s
% RUN: merkc -O2 %S/Inputs/array_noinit_negative.m -o %t && not %t 2>&1 | FileCheck %s --check-prefix=NEGATIVE
% NEGATIVE: Software Error: predicate `array.unsafe_noinit'/2: Unexpected: negative size

:- module array_noinit_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, list, string.

:- type named
    --->    named(string, int).

main(!IO) :-
    strings(Strings),
    io.write_int(array.size(Strings), !IO),
    make(3, Lists : array(list(int))),
    io.write_int(array.size(Lists), !IO),
    named(Named),
    io.write_int(array.size(Named), !IO).

% Each in a procedure of its own, as a procedure reports its first error.
:- pred strings(array(string)::array_uo) is det.
strings(A) :-
% CHECK: :[[@LINE+1]]:5: error: array.unsafe_noinit makes arrays whose elements hold no pointers: ints, uints, floats, chars, enumerations and records of them, not the elements of array.array(string)
    array.unsafe_noinit(2, A).

:- pred named(array(named)::array_uo) is det.
named(A) :-
% CHECK: :[[@LINE+1]]:5: error: array.unsafe_noinit makes arrays whose elements hold no pointers: ints, uints, floats, chars, enumerations and records of them, not the elements of array.array(array_noinit_errors.named)
    array.unsafe_noinit(2, A).

:- pred make(int::in, array(T)::array_uo) is det.
make(N, A) :-
% CHECK: :[[@LINE+1]]:5: error: array.unsafe_noinit makes arrays whose elements hold no pointers, so their type must be known, not a type variable
    array.unsafe_noinit(N, A).
