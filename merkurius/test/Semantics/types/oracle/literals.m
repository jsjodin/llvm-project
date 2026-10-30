% Literals and builtin types: every kind of number, string, character and
% the implementation-defined literals, inferred for undeclared functions
% and predicates. Checked against mmc (utils/type-oracle): the same
% inferred signatures, and merkc's choices, written in as declarations and
% type qualifications, accepted by mmc.
%
% REQUIRES: mmc
% RUN: %compare-types %s

:- module literals.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float, string, char.

main(0).
an_int = 42.
a_negative = -7.
an_underscored = 1_000_000.
a_binary = 0b1010.
an_octal = 0o17.
a_hex = 0xff.
a_char_code = 0'a.
an_int8 = 8i8.
an_int16 = 16i16.
an_int32 = 32i32.
an_int64 = 64i64.
an_explicit_int = 3i.
a_uint = 3u.
a_uint8 = 8u8.
a_uint16 = 16u16.
a_uint32 = 32u32.
a_uint64 = 64u64.
a_float = 3.14.
an_exponent = 1.0e10.
a_string = "abc".
an_escaped_string = "a\nb\"c".
a_char = 'x'.
a_quoted_char = '\n'.
a_line = $line.
a_file = $file.
a_module = $module.
a_pred = $pred.
a_grade = $grade.
a_pair = {1, "a", 'c', 2.0}.
an_empty_tuple = {}.
a_nested_tuple = {{1}, {"a", {}}}.
int_add(X, Y) = X + Y :- X = 1.
float_add(X, Y) = X + Y :- X = 1.0.
string_append(X, Y) = X ++ Y :- X = "a".
int8_mul(X) = X * 2i8.
uint_shift(X) = X << 1 :- X = 1u.
char_to_int(C) = char.to_int(C) + 0.
int_compare(X, Y) :- X < Y, X = 1.
float_compare(X, Y) :- X < Y, Y = 2.0.
