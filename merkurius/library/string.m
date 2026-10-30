% A declaration-only stub of the standard library module string, for name
% resolution until the library is written. The declarations are those of
% the library reference manual (22.01), less those that need modules not
% stubbed yet; see README.md.

:- module string.
:- interface.
:- import_module char.
:- import_module list.

:- func to_char_list(string) = list(char).
:- pred to_char_list(string, list(char)).
:- mode to_char_list(in, out) is det.
:- mode to_char_list(uo, in) is det.
:- func to_rev_char_list(string) = list(char).
:- pred to_rev_char_list(string, list(char)).
:- mode to_rev_char_list(in, out) is det.
:- mode to_rev_char_list(uo, in) is det.
:- func from_char_list(list(char)::in) = (string::uo) is det.
:- pred from_char_list(list(char), string).
:- mode from_char_list(in, uo) is det.
:- mode from_char_list(out, in) is det.
:- func duplicate_char(char::in, int::in) = (string::uo) is det.
:- pred duplicate_char(char::in, int::in, string::uo) is det.
:- pred index(string::in, int::in, char::uo) is semidet.
:- func det_index(string, int) = char.
:- pred det_index(string::in, int::in, char::uo) is det.
:- func length(string::in) = (int::uo) is det.
:- pred length(string, int).
:- mode length(in, uo) is det.
:- mode length(ui, uo) is det.
:- func count_codepoints(string) = int.
:- pred count_codepoints(string::in, int::out) is det.
:- pred is_empty(string::in) is semidet.
:- pred contains_char(string::in, char::in) is semidet.
:- pred sub_string_search(string::in, string::in, int::out) is semidet.
:- func append(string::in, string::in) = (string::uo) is det.
:- pred append(string, string, string).
:- mode append(in, in, in) is semidet.
:- mode append(in, uo, in) is semidet.
:- mode append(in, in, uo) is det.
:- mode append(uo, in, in) is semidet.
:- mode append(out, out, in) is multi.
:- func string ++ string = string.
:- mode in ++ in = uo is det.
:- func append_list(list(string)::in) = (string::uo) is det.
:- pred append_list(list(string)::in, string::uo) is det.
:- func join_list(string::in, list(string)::in) = (string::uo) is det.
:- pred first_char(string, char, string).
:- mode first_char(in, in, in) is semidet.
:- mode first_char(in, uo, in) is semidet.
:- mode first_char(in, in, uo) is semidet.
:- mode first_char(in, uo, uo) is semidet.
:- mode first_char(uo, in, in) is det.
:- func left(string::in, int::in) = (string::out) is det.
:- pred left(string::in, int::in, string::out) is det.
:- func right(string::in, int::in) = (string::out) is det.
:- pred right(string::in, int::in, string::out) is det.
:- func between(string::in, int::in, int::in) = (string::uo) is det.
:- pred between(string::in, int::in, int::in, string::uo) is det.
:- func words(string) = list(string).
:- func split_at_char(char, string) = list(string).
:- func split_into_lines(string) = list(string).
:- pred prefix(string, string).
:- mode prefix(in, in) is semidet.
:- mode prefix(in, out) is multi.
:- pred suffix(string, string).
:- mode suffix(in, in) is semidet.
:- mode suffix(in, out) is multi.
:- pred remove_prefix(string::in, string::in, string::out) is semidet.
:- pred remove_suffix(string::in, string::in, string::out) is semidet.
:- func to_upper(string::in) = (string::uo) is det.
:- pred to_upper(string, string).
:- mode to_upper(in, uo) is det.
:- mode to_upper(in, in) is semidet.
:- func to_lower(string::in) = (string::uo) is det.
:- pred to_lower(string, string).
:- mode to_lower(in, uo) is det.
:- mode to_lower(in, in) is semidet.
:- func pad_left(string, char, int) = string.
:- pred pad_left(string::in, char::in, int::in, string::out) is det.
:- func pad_right(string, char, int) = string.
:- pred pad_right(string::in, char::in, int::in, string::out) is det.
:- func strip(string) = string.
:- func lstrip(string) = string.
:- func rstrip(string) = string.
:- func replace_all(string::in, string::in, string::in) = (string::uo) is det.
:- pred replace_all(string::in, string::in, string::in, string::uo) is det.
:- func foldl(func(char, A) = A, string, A) = A.
:- pred foldl(pred(char, A, A), string, A, A).
:- mode foldl(pred(in, di, uo) is det, in, di, uo) is det.
:- mode foldl(pred(in, in, out) is det, in, in, out) is det.
:- mode foldl(pred(in, in, out) is semidet, in, in, out) is semidet.
:- mode foldl(pred(in, in, out) is nondet, in, in, out) is nondet.
:- mode foldl(pred(in, in, out) is multi, in, in, out) is multi.
:- pred to_int(string::in, int::out) is semidet.
:- func det_to_int(string) = int.
:- pred to_float(string::in, float::out) is semidet.
:- func det_to_float(string) = float.
:- func char_to_string(char::in) = (string::uo) is det.
:- pred char_to_string(char, string).
:- mode char_to_string(in, uo) is det.
:- mode char_to_string(out, in) is semidet.
:- func from_char(char::in) = (string::uo) is det.
:- func int_to_string(int::in) = (string::uo) is det.
:- pred int_to_string(int::in, string::uo) is det.
:- func from_int(int::in) = (string::uo) is det.
:- func float_to_string(float::in) = (string::uo) is det.
:- pred float_to_string(float::in, string::uo) is det.
:- func from_float(float::in) = (string::uo) is det.
:- func string(T) = string.
:- type poly_type
    --->    f(float)
    ;       i(int)
    ;       i8(int8)
    ;       i16(int16)
    ;       i32(int32)
    ;       i64(int64)
    ;       u(uint)
    ;       u8(uint8)
    ;       u16(uint16)
    ;       u32(uint32)
    ;       u64(uint64)
    ;       s(string)
    ;       c(char).
:- func format(string, list(poly_type)) = string.
:- pred format(string::in, list(poly_type)::in, string::out) is det.
