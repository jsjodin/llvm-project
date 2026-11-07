% The standard library module char: Unicode code points, their classes in
% ASCII and in Unicode 13, digits in bases up to 36, and their encodings in
% UTF-8 and UTF-16. The declarations are those of the library reference
% manual (22.01), less those that need modules not stubbed yet; see
% README.md. What the manual leaves open, such as the errors' messages,
% which code points from_int accepts, and hash, is mmc's, found by running
% its programs.

:- module char.
:- interface.
:- import_module list.

:- type char == character.
:- func to_int(char) = int.
:- pred to_int(char, int).
:- mode to_int(in, out) is det.
:- mode to_int(in, in) is semidet.
:- mode to_int(out, in) is semidet.
:- pred from_int(int::in, char::out) is semidet.
:- func det_from_int(int) = char.
:- pred det_from_int(int::in, char::out) is det.
:- func min_char_value = int.
:- pred min_char_value(int::out) is det.
:- func max_char_value = int.
:- pred max_char_value(int::out) is det.
:- pred is_lower(char::in) is semidet.
:- pred is_upper(char::in) is semidet.
:- func to_lower(char) = char.
:- pred to_lower(char::in, char::out) is det.
:- func to_upper(char) = char.
:- pred to_upper(char::in, char::out) is det.
:- pred lower_upper(char, char).
:- mode lower_upper(in, out) is semidet.
:- mode lower_upper(out, in) is semidet.
:- pred is_ascii(char::in) is semidet.
:- pred is_whitespace(char::in) is semidet.
:- pred is_alpha(char::in) is semidet.
:- pred is_alnum(char::in) is semidet.
:- pred is_alpha_or_underscore(char::in) is semidet.
:- pred is_alnum_or_underscore(char::in) is semidet.
:- pred is_digit(char::in) is semidet.
:- pred is_binary_digit(char::in) is semidet.
:- pred is_octal_digit(char::in) is semidet.
:- pred is_decimal_digit(char::in) is semidet.
:- pred is_hex_digit(char::in) is semidet.
:- pred is_base_digit(int::in, char::in) is semidet.
:- pred binary_digit_to_int(char::in, int::out) is semidet.
:- func det_binary_digit_to_int(char) = int.
:- pred octal_digit_to_int(char::in, int::out) is semidet.
:- func det_octal_digit_to_int(char) = int.
:- pred decimal_digit_to_int(char::in, int::out) is semidet.
:- func det_decimal_digit_to_int(char) = int.
:- pred hex_digit_to_int(char::in, int::out) is semidet.
:- func det_hex_digit_to_int(char) = int.
:- pred base_digit_to_int(int::in, char::in, int::out) is semidet.
:- func det_base_digit_to_int(int, char) = int.
:- pred unsafe_base_digit_to_int(int::in, char::in, int::out) is semidet.
:- pred int_to_binary_digit(int::in, char::out) is semidet.
:- func det_int_to_binary_digit(int) = char.
:- pred int_to_octal_digit(int::in, char::out) is semidet.
:- func det_int_to_octal_digit(int) = char.
:- pred int_to_decimal_digit(int::in, char::out) is semidet.
:- func det_int_to_decimal_digit(int) = char.
:- pred int_to_hex_digit(int::in, char::out) is semidet.
:- func det_int_to_hex_digit(int) = char.
:- pred base_int_to_digit(int::in, int::in, char::out) is semidet.
:- func det_base_int_to_digit(int, int) = char.
:- pred to_utf8(char::in, list(int)::out) is semidet.
:- pred to_utf8_uint8(char::in, list(uint8)::out) is semidet.
:- pred to_utf16(char::in, list(int)::out) is semidet.
:- pred to_utf16_uint16(char::in, list(uint16)::out) is semidet.
:- pred is_surrogate(char::in) is semidet.
:- pred is_leading_surrogate(char::in) is semidet.
:- pred is_trailing_surrogate(char::in) is semidet.
:- pred is_noncharacter(char::in) is semidet.
:- pred is_control(char::in) is semidet.
:- pred is_space_separator(char::in) is semidet.
:- pred is_line_separator(char::in) is semidet.
:- pred is_paragraph_separator(char::in) is semidet.
:- pred is_private_use(char::in) is semidet.
:- pred is_hex_digit(char, int).
:- mode is_hex_digit(in, out) is semidet.
:- pred int_to_hex_char(int, char).
:- mode int_to_hex_char(in, out) is semidet.
:- func hash(char) = int.
:- pred hash(char::in, int::out) is det.

%---------------------------------------------------------------------------%

:- implementation.
:- import_module int.
:- import_module require.
:- import_module uint16.
:- import_module uint64.
:- import_module uint8.

% A character is its code point.

:- pragma foreign_proc("C",
    to_int(Char::in) = (Code::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Code = (MR_Integer) Char;
").

:- pragma foreign_proc("C",
    to_int(Char::in, Code::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Code = (MR_Integer) Char;
").

:- pragma foreign_proc("C",
    to_int(Char::in, Code::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = ((MR_Integer) Char == Code);
").

    % Surrogates and noncharacters are code points too. As mmc's, only the
    % int's low 32 bits count, so 0x100000041 is 'A' (design document,
    % "Where mmc and the manual disagree").
:- pragma foreign_proc("C",
    to_int(Char::out, Code::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    uint32_t low = (uint32_t) Code;
    Char = (MR_Char) low;
    SUCCESS_INDICATOR = (low <= 0x10ffff);
").

from_int(Code, Char) :-
    to_int(Char, Code).

det_from_int(Code) = Char :-
    det_from_int(Code, Char).

det_from_int(Code, Char) :-
    ( if from_int(Code, Char0) then
        Char = Char0
    else
        unexpected("predicate `char.det_from_int'/2", "conversion failed")
    ).

min_char_value = 0.

min_char_value(0).

max_char_value = 0x10ffff.

max_char_value(0x10ffff).

%---------------------------------------------------------------------------%

is_lower(Char) :-
    Code = to_int(Char),
    Code >= 0x61,
    Code =< 0x7a.

is_upper(Char) :-
    Code = to_int(Char),
    Code >= 0x41,
    Code =< 0x5a.

to_lower(Char) = Lower :-
    to_lower(Char, Lower).

to_lower(Char, Lower) :-
    ( if is_upper(Char) then
        Lower = det_from_int(to_int(Char) + 0x20)
    else
        Lower = Char
    ).

to_upper(Char) = Upper :-
    to_upper(Char, Upper).

to_upper(Char, Upper) :-
    ( if is_lower(Char) then
        Upper = det_from_int(to_int(Char) - 0x20)
    else
        Upper = Char
    ).

:- pragma promise_equivalent_clauses(pred(lower_upper/2)).

lower_upper(Lower::in, Upper::out) :-
    is_lower(Lower),
    Upper = det_from_int(to_int(Lower) - 0x20).
lower_upper(Lower::out, Upper::in) :-
    is_upper(Upper),
    Lower = det_from_int(to_int(Upper) + 0x20).

%---------------------------------------------------------------------------%

is_ascii(Char) :-
    to_int(Char) < 0x80.

is_whitespace(Char) :-
    Code = to_int(Char),
    ( Code = 0x20
    ; Code >= 0x09, Code =< 0x0d
    ).

is_alpha(Char) :-
    ( is_lower(Char)
    ; is_upper(Char)
    ).

is_alnum(Char) :-
    ( is_alpha(Char)
    ; is_digit(Char)
    ).

is_alpha_or_underscore(Char) :-
    ( is_alpha(Char)
    ; Char = '_'
    ).

is_alnum_or_underscore(Char) :-
    ( is_alnum(Char)
    ; Char = '_'
    ).

%---------------------------------------------------------------------------%

is_digit(Char) :-
    Code = to_int(Char),
    Code >= 0x30,
    Code =< 0x39.

is_binary_digit(Char) :-
    binary_digit_to_int(Char, _).

is_octal_digit(Char) :-
    octal_digit_to_int(Char, _).

is_decimal_digit(Char) :-
    is_digit(Char).

is_hex_digit(Char) :-
    hex_digit_to_int(Char, _).

is_base_digit(Base, Char) :-
    ( if Base < 2 ; Base > 36 then
        error("predicate `char.is_base_digit'/2", "invalid base")
    else
        unsafe_base_digit_to_int(Base, Char, _)
    ).

%---------------------------------------------------------------------------%

    % The value of a digit, a letter of either case counting from 10, or
    % fails for any other character.
:- pred any_digit_to_int(char::in, int::out) is semidet.

any_digit_to_int(Char, Value) :-
    Code = to_int(Char),
    ( if Code >= 0x30, Code =< 0x39 then
        Value = Code - 0x30
    else if Code >= 0x61, Code =< 0x7a then
        Value = Code - 0x61 + 10
    else
        Code >= 0x41,
        Code =< 0x5a,
        Value = Code - 0x41 + 10
    ).

binary_digit_to_int(Char, Value) :-
    unsafe_base_digit_to_int(2, Char, Value).

det_binary_digit_to_int(Char) = Value :-
    ( if binary_digit_to_int(Char, Value0) then
        Value = Value0
    else
        error("function `char.det_binary_digit_to_int'/1",
            "char.binary_digit_to_int failed")
    ).

octal_digit_to_int(Char, Value) :-
    unsafe_base_digit_to_int(8, Char, Value).

det_octal_digit_to_int(Char) = Value :-
    ( if octal_digit_to_int(Char, Value0) then
        Value = Value0
    else
        error("function `char.det_octal_digit_to_int'/1",
            "char.octal_digit_to_int failed")
    ).

decimal_digit_to_int(Char, Value) :-
    unsafe_base_digit_to_int(10, Char, Value).

det_decimal_digit_to_int(Char) = Value :-
    ( if decimal_digit_to_int(Char, Value0) then
        Value = Value0
    else
        error("function `char.det_decimal_digit_to_int'/1",
            "char.decimal_digit_to_int failed")
    ).

hex_digit_to_int(Char, Value) :-
    unsafe_base_digit_to_int(16, Char, Value).

det_hex_digit_to_int(Char) = Value :-
    ( if hex_digit_to_int(Char, Value0) then
        Value = Value0
    else
        error("function `char.det_hex_digit_to_int'/1",
            "char.hex_digit_to_int failed")
    ).

base_digit_to_int(Base, Char, Value) :-
    ( if Base < 2 ; Base > 36 then
        error("predicate `char.base_digit_to_int'/3",
            "base is not in the range 2 .. 36")
    else
        unsafe_base_digit_to_int(Base, Char, Value)
    ).

det_base_digit_to_int(Base, Char) = Value :-
    ( if base_digit_to_int(Base, Char, Value0) then
        Value = Value0
    else
        error("function `char.det_base_digit_to_int'/2",
            "char.base_digit_to_int failed")
    ).

unsafe_base_digit_to_int(Base, Char, Value) :-
    any_digit_to_int(Char, Value),
    Value < Base.

%---------------------------------------------------------------------------%

    % The digit of a value from 0 to 35, with uppercase letters from 10.
:- func digit(int) = char.

digit(Value) =
    det_from_int(( if Value < 10 then 0x30 + Value else 0x41 + Value - 10 )).

int_to_binary_digit(Value, Char) :-
    Value >= 0,
    Value < 2,
    Char = digit(Value).

det_int_to_binary_digit(Value) = Char :-
    ( if int_to_binary_digit(Value, Char0) then
        Char = Char0
    else
        error("function `char.det_int_to_binary_digit'/1",
            "char.int_to_binary_digit failed")
    ).

int_to_octal_digit(Value, Char) :-
    Value >= 0,
    Value < 8,
    Char = digit(Value).

det_int_to_octal_digit(Value) = Char :-
    ( if int_to_octal_digit(Value, Char0) then
        Char = Char0
    else
        error("function `char.det_int_to_octal_digit'/1",
            "char.int_to_octal_digit failed")
    ).

int_to_decimal_digit(Value, Char) :-
    Value >= 0,
    Value < 10,
    Char = digit(Value).

det_int_to_decimal_digit(Value) = Char :-
    ( if int_to_decimal_digit(Value, Char0) then
        Char = Char0
    else
        error("function `char.det_int_to_decimal_digit'/1",
            "char.int_to_decimal_digit failed")
    ).

int_to_hex_digit(Value, Char) :-
    Value >= 0,
    Value < 16,
    Char = digit(Value).

det_int_to_hex_digit(Value) = Char :-
    ( if int_to_hex_digit(Value, Char0) then
        Char = Char0
    else
        error("function `char.det_int_to_hex_digit'/1",
            "char.int_to_hex_digit failed")
    ).

base_int_to_digit(Base, Value, Char) :-
    ( if Base < 2 ; Base > 36 then
        error("predicate `char.base_int_to_digit'/3", "invalid base")
    else
        Value >= 0,
        Value < Base,
        Char = digit(Value)
    ).

det_base_int_to_digit(Base, Value) = Char :-
    ( if base_int_to_digit(Base, Value, Char0) then
        Char = Char0
    else
        error("function `char.det_base_int_to_digit'/2",
            "char.base_int_to_digit failed")
    ).

%---------------------------------------------------------------------------%

to_utf8(Char, Units) :-
    Code = to_int(Char),
    not is_surrogate(Char),
    ( if Code < 0x80 then
        Units = [Code]
    else if Code < 0x800 then
        Units = [0xc0 \/ (Code >> 6), 0x80 \/ (Code /\ 0x3f)]
    else if Code < 0x10000 then
        Units = [0xe0 \/ (Code >> 12), 0x80 \/ ((Code >> 6) /\ 0x3f),
            0x80 \/ (Code /\ 0x3f)]
    else
        Units = [0xf0 \/ (Code >> 18), 0x80 \/ ((Code >> 12) /\ 0x3f),
            0x80 \/ ((Code >> 6) /\ 0x3f), 0x80 \/ (Code /\ 0x3f)]
    ).

to_utf8_uint8(Char, Units) :-
    to_utf8(Char, Ints),
    Units = to_uint8s(Ints).

:- func to_uint8s(list(int)) = list(uint8).

to_uint8s([]) = [].
to_uint8s([I | Is]) = [uint8.cast_from_int(I) | to_uint8s(Is)].

to_utf16(Char, Units) :-
    Code = to_int(Char),
    not is_surrogate(Char),
    ( if Code < 0x10000 then
        Units = [Code]
    else
        Offset = Code - 0x10000,
        Units = [0xd800 \/ (Offset >> 10), 0xdc00 \/ (Offset /\ 0x3ff)]
    ).

to_utf16_uint16(Char, Units) :-
    to_utf16(Char, Ints),
    Units = to_uint16s(Ints).

:- func to_uint16s(list(int)) = list(uint16).

to_uint16s([]) = [].
to_uint16s([I | Is]) = [uint16.cast_from_int(I) | to_uint16s(Is)].

%---------------------------------------------------------------------------%
% Unicode 13's classes.

is_surrogate(Char) :-
    Code = to_int(Char),
    Code >= 0xd800,
    Code =< 0xdfff.

is_leading_surrogate(Char) :-
    Code = to_int(Char),
    Code >= 0xd800,
    Code =< 0xdbff.

is_trailing_surrogate(Char) :-
    Code = to_int(Char),
    Code >= 0xdc00,
    Code =< 0xdfff.

    % U+FDD0 to U+FDEF, and the last two code points of each plane.
is_noncharacter(Char) :-
    Code = to_int(Char),
    ( Code >= 0xfdd0, Code =< 0xfdef
    ; Code /\ 0xfffe = 0xfffe
    ).

    % General category Cc.
is_control(Char) :-
    Code = to_int(Char),
    ( Code =< 0x1f
    ; Code >= 0x7f, Code =< 0x9f
    ).

    % General category Zs.
is_space_separator(Char) :-
    Code = to_int(Char),
    ( Code = 0x20
    ; Code = 0xa0
    ; Code = 0x1680
    ; Code >= 0x2000, Code =< 0x200a
    ; Code = 0x202f
    ; Code = 0x205f
    ; Code = 0x3000
    ).

is_line_separator(Char) :-
    to_int(Char) = 0x2028.

is_paragraph_separator(Char) :-
    to_int(Char) = 0x2029.

    % General category Co: the private use area and planes 15 and 16, less
    % their noncharacters.
is_private_use(Char) :-
    Code = to_int(Char),
    ( Code >= 0xe000, Code =< 0xf8ff
    ; Code >= 0xf0000, Code =< 0xffffd
    ; Code >= 0x100000, Code =< 0x10fffd
    ).

%---------------------------------------------------------------------------%

is_hex_digit(Char, Value) :-
    hex_digit_to_int(Char, Value).

int_to_hex_char(Value, Char) :-
    int_to_hex_digit(Value, Char).

%---------------------------------------------------------------------------%

    % Thomas Wang's 64-bit integer hash of the code point, as mmc's.
hash(Char) = Hash :-
    hash(Char, Hash).

hash(Char, Hash) :-
    Key0 = uint64.cast_from_int(to_int(Char)),
    Key1 = \ Key0 + (Key0 << 21),
    Key2 = Key1 `xor` (Key1 >> 24),
    Key3 = Key2 + (Key2 << 3) + (Key2 << 8),
    Key4 = Key3 `xor` (Key3 >> 14),
    Key5 = Key4 + (Key4 << 2) + (Key4 << 4),
    Key6 = Key5 `xor` (Key5 >> 28),
    Key7 = Key6 + (Key6 << 31),
    Hash = uint64.cast_to_int(Key7).
