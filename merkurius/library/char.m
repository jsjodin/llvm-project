% A declaration-only stub of the standard library module char, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

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
