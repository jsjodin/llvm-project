% The char module, every declaration and mode, against mmc's output:
% conversions to and from code points, which keep an int's low 32 bits as
% mmc's do (design document, "Where mmc and the manual disagree"); the
% ASCII classes, case and digits in each base; the Unicode classes, each as
% the ranges of all code points for which it holds; UTF-8 and UTF-16
% encodings at each length's edges and of surrogates; and hash. The
% exceptions of the det_ functions are char-errors.m's.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:{97, 1114111, 0}
% OUTPUT-NEXT:"fails"
% OUTPUT-NEXT:0
% OUTPUT-NEXT:55296
% OUTPUT-NEXT:57343
% OUTPUT-NEXT:65534
% OUTPUT-NEXT:1114111
% OUTPUT-NEXT:"fails"
% OUTPUT-NEXT:"fails"
% OUTPUT-NEXT:0
% OUTPUT-NEXT:65
% OUTPUT-NEXT:65
% OUTPUT-NEXT:"fails"
% OUTPUT-NEXT:"fails"
% OUTPUT-NEXT:'A'
% OUTPUT-NEXT:"no"
% OUTPUT-NEXT:"yes"
% OUTPUT-NEXT:"no"
% OUTPUT-NEXT:{0, 1114111}
% OUTPUT-NEXT:{0, 1114111}
% OUTPUT-NEXT:['a', 'z', 'é', '1', 'z', 'a', '@', '[', '`', '{', 'É', 'ÿ']
% OUTPUT-NEXT:['A', 'Z', 'é', '1', 'Z', 'A', '@', '[', '`', '{', 'É', 'ÿ']
% OUTPUT-NEXT:['Z', 'A']
% OUTPUT-NEXT:['a', 'z']
% OUTPUT-NEXT:is_lower: [{97, 122}]
% OUTPUT-NEXT:is_upper: [{65, 90}]
% OUTPUT-NEXT:is_ascii: [{0, 127}]
% OUTPUT-NEXT:is_whitespace: [{9, 13}, {32, 32}]
% OUTPUT-NEXT:is_alpha: [{65, 90}, {97, 122}]
% OUTPUT-NEXT:is_alnum: [{48, 57}, {65, 90}, {97, 122}]
% OUTPUT-NEXT:is_alpha_or_underscore: [{65, 90}, {95, 95}, {97, 122}]
% OUTPUT-NEXT:is_alnum_or_underscore: [{48, 57}, {65, 90}, {95, 95}, {97, 122}]
% OUTPUT-NEXT:is_digit: [{48, 57}]
% OUTPUT-NEXT:is_binary_digit: [{48, 49}]
% OUTPUT-NEXT:is_octal_digit: [{48, 55}]
% OUTPUT-NEXT:is_decimal_digit: [{48, 57}]
% OUTPUT-NEXT:is_hex_digit: [{48, 57}, {65, 70}, {97, 102}]
% OUTPUT-NEXT:is_base_digit 2: [{48, 49}]
% OUTPUT-NEXT:is_base_digit 16: [{48, 57}, {65, 70}, {97, 102}]
% OUTPUT-NEXT:is_base_digit 36: [{48, 57}, {65, 90}, {97, 122}]
% OUTPUT-NEXT:is_surrogate: [{55296, 57343}]
% OUTPUT-NEXT:is_leading_surrogate: [{55296, 56319}]
% OUTPUT-NEXT:is_trailing_surrogate: [{56320, 57343}]
% OUTPUT-NEXT:is_noncharacter: [{64976, 65007}, {65534, 65535}, {131070, 131071}, {196606, 196607}, {262142, 262143}, {327678, 327679}, {393214, 393215}, {458750, 458751}, {524286, 524287}, {589822, 589823}, {655358, 655359}, {720894, 720895}, {786430, 786431}, {851966, 851967}, {917502, 917503}, {983038, 983039}, {1048574, 1048575}, {1114110, 1114111}]
% OUTPUT-NEXT:is_control: [{0, 31}, {127, 159}]
% OUTPUT-NEXT:is_space_separator: [{32, 32}, {160, 160}, {5760, 5760}, {8192, 8202}, {8239, 8239}, {8287, 8287}, {12288, 12288}]
% OUTPUT-NEXT:is_line_separator: [{8232, 8232}]
% OUTPUT-NEXT:is_paragraph_separator: [{8233, 8233}]
% OUTPUT-NEXT:is_private_use: [{57344, 63743}, {983040, 1048573}, {1048576, 1114109}]
% OUTPUT-NEXT:[{0, 0, 0, 0}, {1, 1, 1, 1}, {-1, 2, 2, 2}, {-1, 7, 7, 7}, {-1, -1, 8, 8}, {-1, -1, 9, 9}, {-1, -1, -1, 10}, {-1, -1, -1, 15}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, 10}, {-1, -1, -1, 15}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}]
% OUTPUT-NEXT:[{0, 0, 0, 0}, {1, 1, 1, 1}, {-1, 2, 2, 2}, {-1, 7, 7, 7}, {-1, 8, 8, 8}, {-1, 9, 9, 9}, {-1, -1, 10, 10}, {-1, -1, 15, 15}, {-1, -1, -1, 16}, {-1, -1, -1, 35}, {-1, -1, 10, 10}, {-1, -1, 15, 15}, {-1, -1, -1, 16}, {-1, -1, -1, 35}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}]
% OUTPUT-NEXT:[{"-", "-", "-", "-"}, {"0", "0", "0", "0"}, {"1", "1", "1", "1"}, {"-", "2", "2", "2"}, {"-", "3", "3", "3"}, {"-", "4", "4", "4"}, {"-", "5", "5", "5"}, {"-", "6", "6", "6"}, {"-", "7", "7", "7"}, {"-", "-", "8", "8"}, {"-", "-", "9", "9"}, {"-", "-", "-", "A"}, {"-", "-", "-", "B"}, {"-", "-", "-", "C"}, {"-", "-", "-", "D"}, {"-", "-", "-", "E"}, {"-", "-", "-", "F"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}, {"-", "-", "-", "-"}]
% OUTPUT-NEXT:[{"-", "-", "-"}, {"0", "0", "0"}, {"1", "1", "1"}, {"-", "2", "2"}, {"-", "3", "3"}, {"-", "4", "4"}, {"-", "5", "5"}, {"-", "6", "6"}, {"-", "7", "7"}, {"-", "8", "8"}, {"-", "9", "9"}, {"-", "A", "A"}, {"-", "B", "B"}, {"-", "C", "C"}, {"-", "D", "D"}, {"-", "E", "E"}, {"-", "F", "F"}, {"-", "-", "G"}, {"-", "-", "H"}, {"-", "-", "I"}, {"-", "-", "J"}, {"-", "-", "K"}, {"-", "-", "L"}, {"-", "-", "M"}, {"-", "-", "N"}, {"-", "-", "O"}, {"-", "-", "P"}, {"-", "-", "Q"}, {"-", "-", "R"}, {"-", "-", "S"}, {"-", "-", "T"}, {"-", "-", "U"}, {"-", "-", "V"}, {"-", "-", "W"}, {"-", "-", "X"}, {"-", "-", "Y"}, {"-", "-", "Z"}, {"-", "-", "-"}, {"-", "-", "-"}]
% OUTPUT-NEXT:{0, "[0]", "[0u8]", "[0]", "[0u16]"}
% OUTPUT-NEXT:{127, "[127]", "[127u8]", "[127]", "[127u16]"}
% OUTPUT-NEXT:{128, "[194, 128]", "[194u8, 128u8]", "[128]", "[128u16]"}
% OUTPUT-NEXT:{2047, "[223, 191]", "[223u8, 191u8]", "[2047]", "[2047u16]"}
% OUTPUT-NEXT:{2048, "[224, 160, 128]", "[224u8, 160u8, 128u8]", "[2048]", "[2048u16]"}
% OUTPUT-NEXT:{65535, "[239, 191, 191]", "[239u8, 191u8, 191u8]", "[65535]", "[65535u16]"}
% OUTPUT-NEXT:{65536, "[240, 144, 128, 128]", "[240u8, 144u8, 128u8, 128u8]", "[55296, 56320]", "[55296u16, 56320u16]"}
% OUTPUT-NEXT:{1114111, "[244, 143, 191, 191]", "[244u8, 143u8, 191u8, 191u8]", "[56319, 57343]", "[56319u16, 57343u16]"}
% OUTPUT-NEXT:{55296, "fails", "fails", "fails", "fails"}
% OUTPUT-NEXT:{56319, "fails", "fails", "fails", "fails"}
% OUTPUT-NEXT:{56320, "fails", "fails", "fails", "fails"}
% OUTPUT-NEXT:{57343, "fails", "fails", "fails", "fails"}
% OUTPUT-NEXT:{65534, "[239, 191, 190]", "[239u8, 191u8, 190u8]", "[65534]", "[65534u16]"}
% OUTPUT-NEXT:{233, "[195, 169]", "[195u8, 169u8]", "[233]", "[233u16]"}
% OUTPUT-NEXT:[-3864037893810923019, 2561069026051980119, -7836329298054118631, 8633297058295171728, -3859784353626049694]
% OUTPUT-NEXT:-3864037893810923019

:- module char_library.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, int, list, string, uint8, uint16.

main(!IO) :-
    part0(!IO),
    part1(!IO),
    part2(!IO),
    part3(!IO),
    part4(!IO),
    part5(!IO),
    part6(!IO),
    part7(!IO),
    part8(!IO),
    part9(!IO),
    part10(!IO).

:- pred part0(io::di, io::uo) is det.
part0(!IO) :-
    io.write_line({code('a'), code(char.det_from_int(0x10ffff)), code(char.det_from_int(0))}, !IO),
    list.foldl(show_from_int, [-1, 0, 0xd800, 0xdfff, 0xfffe, 0x10ffff, 0x110000, max_int, min_int, 0x100000041, min_int + 0x41, 0x80000041, -0x80000000], !IO),
    ( if char.to_int(C1, 65) then io.write_line(C1, !IO) else io.write_line("no", !IO) ),
    ( if char.to_int(_, -5) then io.write_line("yes", !IO) else io.write_line("no", !IO) ).

:- pred part1(io::di, io::uo) is det.
part1(!IO) :-
    ( if char.to_int('A', 65) then io.write_line("yes", !IO) else io.write_line("no", !IO) ),
    ( if char.to_int('A', 66) then io.write_line("yes", !IO) else io.write_line("no", !IO) ),
    io.write_line({char.min_char_value : int, char.max_char_value : int}, !IO),
    char.min_char_value(Min), char.max_char_value(Max), io.write_line({Min, Max}, !IO).

:- pred part2(io::di, io::uo) is det.
part2(!IO) :-
    io.write_line(list.map(lower, samples) : list(char), !IO),
    io.write_line(list.map(upper, samples) : list(char), !IO),
    list.filter_map(lu1, samples, Kept1), io.write_line(Kept1, !IO),
    list.filter_map(lu2, samples, Kept2), io.write_line(Kept2, !IO).

:- pred part3(io::di, io::uo) is det.
part3(!IO) :-
    ranges("is_lower", (pred(C::in) is semidet :- char.is_lower(C)), !IO),
    ranges("is_upper", (pred(C::in) is semidet :- char.is_upper(C)), !IO),
    ranges("is_ascii", (pred(C::in) is semidet :- char.is_ascii(C)), !IO),
    ranges("is_whitespace", (pred(C::in) is semidet :- char.is_whitespace(C)), !IO).

:- pred part4(io::di, io::uo) is det.
part4(!IO) :-
    ranges("is_alpha", (pred(C::in) is semidet :- char.is_alpha(C)), !IO),
    ranges("is_alnum", (pred(C::in) is semidet :- char.is_alnum(C)), !IO),
    ranges("is_alpha_or_underscore", (pred(C::in) is semidet :- char.is_alpha_or_underscore(C)), !IO),
    ranges("is_alnum_or_underscore", (pred(C::in) is semidet :- char.is_alnum_or_underscore(C)), !IO).

:- pred part5(io::di, io::uo) is det.
part5(!IO) :-
    ranges("is_digit", (pred(C::in) is semidet :- char.is_digit(C)), !IO),
    ranges("is_binary_digit", (pred(C::in) is semidet :- char.is_binary_digit(C)), !IO),
    ranges("is_octal_digit", (pred(C::in) is semidet :- char.is_octal_digit(C)), !IO),
    ranges("is_decimal_digit", (pred(C::in) is semidet :- char.is_decimal_digit(C)), !IO).

:- pred part6(io::di, io::uo) is det.
part6(!IO) :-
    ranges("is_hex_digit", (pred(C::in) is semidet :- char.is_hex_digit(C)), !IO),
    ranges("is_base_digit 2", (pred(C::in) is semidet :- char.is_base_digit(2, C)), !IO),
    ranges("is_base_digit 16", (pred(C::in) is semidet :- char.is_base_digit(16, C)), !IO),
    ranges("is_base_digit 36", (pred(C::in) is semidet :- char.is_base_digit(36, C)), !IO).

:- pred part7(io::di, io::uo) is det.
part7(!IO) :-
    ranges("is_surrogate", (pred(C::in) is semidet :- char.is_surrogate(C)), !IO),
    ranges("is_leading_surrogate", (pred(C::in) is semidet :- char.is_leading_surrogate(C)), !IO),
    ranges("is_trailing_surrogate", (pred(C::in) is semidet :- char.is_trailing_surrogate(C)), !IO),
    ranges("is_noncharacter", (pred(C::in) is semidet :- char.is_noncharacter(C)), !IO).

:- pred part8(io::di, io::uo) is det.
part8(!IO) :-
    ranges("is_control", (pred(C::in) is semidet :- char.is_control(C)), !IO),
    ranges("is_space_separator", (pred(C::in) is semidet :- char.is_space_separator(C)), !IO),
    ranges("is_line_separator", (pred(C::in) is semidet :- char.is_line_separator(C)), !IO),
    ranges("is_paragraph_separator", (pred(C::in) is semidet :- char.is_paragraph_separator(C)), !IO).

:- pred part9(io::di, io::uo) is det.
part9(!IO) :-
    ranges("is_private_use", (pred(C::in) is semidet :- char.is_private_use(C)), !IO),
    io.write_line(list.map(digit_values, digits) : list({int, int, int, int}), !IO),
    io.write_line(list.map(base_values, digits) : list({int, int, int, int}), !IO),
    io.write_line(list.map(int_digits, -1 .. 37) : list({string, string, string, string}), !IO).

:- pred part10(io::di, io::uo) is det.
part10(!IO) :-
    io.write_line(list.map(base_int_digits, -1 .. 37) : list({string, string, string}), !IO),
    list.foldl(show_utf, [0, 0x7f, 0x80, 0x7ff, 0x800, 0xffff, 0x10000, 0x10ffff, 0xd800, 0xdbff, 0xdc00, 0xdfff, 0xfffe, 0xe9], !IO),
    io.write_line(list.map(hash_of, ['a', 'b', 'é', char.det_from_int(0), char.det_from_int(0x10ffff)]) : list(int), !IO),
    char.hash('a', HashA), io.write_line(HashA, !IO).








:- func samples = list(char).
samples = ['A', 'z', 'é', '1', 'Z', 'a', '@', '[', '`', '{', 'É', 'ÿ'].

:- func digits = list(char).
digits = ['0', '1', '2', '7', '8', '9', 'a', 'f', 'g', 'z', 'A', 'F', 'G', 'Z', '/', ':', '@', '[', '`', '{', 'é'].


:- pred lu1(char::in, char::out) is semidet.
lu1(L, U) :- char.lower_upper(L, U).
:- pred lu2(char::in, char::out) is semidet.
lu2(U, L) :- char.lower_upper(L, U).

:- pred show_from_int(int::in, io::di, io::uo) is det.
show_from_int(I, !IO) :-
    ( if char.from_int(I, C) then io.write_line(code(C), !IO) else io.write_line("fails", !IO) ).

:- pred ranges(string::in, pred(char)::in(pred(in) is semidet), io::di, io::uo) is det.
ranges(Name, P, !IO) :-
    collect(P, 0, -1, [], Rs),
    io.write_string(Name ++ ": ", !IO),
    io.write_line(list.reverse(Rs) : list({int, int}), !IO).

:- pred collect(pred(char)::in(pred(in) is semidet), int::in, int::in, list({int, int})::in, list({int, int})::out) is det.
collect(P, I, Start, !Rs) :-
    ( if I > 0x10ffff then
        ( if Start >= 0 then !:Rs = [{Start, I - 1} | !.Rs] else true )
    else
        ( if P(char.det_from_int(I)) then
            ( if Start >= 0 then S = Start else S = I ),
            collect(P, I + 1, S, !Rs)
        else
            ( if Start >= 0 then !:Rs = [{Start, I - 1} | !.Rs] else true ),
            collect(P, I + 1, -1, !Rs)
        )
    ).

:- func digit_values(char) = {int, int, int, int}.
digit_values(C) = {B, O, D, H} :-
    ( if char.binary_digit_to_int(C, B0) then B = B0 else B = -1 ),
    ( if char.octal_digit_to_int(C, O0) then O = O0 else O = -1 ),
    ( if char.decimal_digit_to_int(C, D0) then D = D0 else D = -1 ),
    ( if char.hex_digit_to_int(C, H0) then H = H0 else H = -1 ).

:- func base_values(char) = {int, int, int, int}.
base_values(C) = {B, T, H, F} :-
    ( if char.base_digit_to_int(2, C, B0) then B = B0 else B = -1 ),
    ( if char.base_digit_to_int(10, C, T0) then T = T0 else T = -1 ),
    ( if char.base_digit_to_int(16, C, H0) then H = H0 else H = -1 ),
    ( if char.base_digit_to_int(36, C, F0) then F = F0 else F = -1 ).

:- func int_digits(int) = {string, string, string, string}.
int_digits(I) = {B, O, D, H} :-
    ( if char.int_to_binary_digit(I, B0) then B = char_to_string(B0) else B = "-" ),
    ( if char.int_to_octal_digit(I, O0) then O = char_to_string(O0) else O = "-" ),
    ( if char.int_to_decimal_digit(I, D0) then D = char_to_string(D0) else D = "-" ),
    ( if char.int_to_hex_digit(I, H0) then H = char_to_string(H0) else H = "-" ).

:- func base_int_digits(int) = {string, string, string}.
base_int_digits(I) = {B, H, F} :-
    ( if char.base_int_to_digit(2, I, B0) then B = char_to_string(B0) else B = "-" ),
    ( if char.base_int_to_digit(16, I, H0) then H = char_to_string(H0) else H = "-" ),
    ( if char.base_int_to_digit(36, I, F0) then F = char_to_string(F0) else F = "-" ).

:- pred show_utf(int::in, io::di, io::uo) is det.
show_utf(I, !IO) :-
    C = char.det_from_int(I),
    ( if char.to_utf8(C, U8) then A = string(U8) else A = "fails" ),
    ( if char.to_utf8_uint8(C, U8b) then B = string(U8b) else B = "fails" ),
    ( if char.to_utf16(C, U16) then D = string(U16) else D = "fails" ),
    ( if char.to_utf16_uint16(C, U16b) then E = string(U16b) else E = "fails" ),
    io.write_line({I, A, B, D, E}, !IO).

:- pred old_hex(char::in, int::out) is semidet.
old_hex(C, I) :- char.is_hex_digit(C, I).
:- pred old_hex_char(int::in, char::out) is semidet.
old_hex_char(I, C) :- char.int_to_hex_char(I, C).
:- func code(char) = int.
code(C) = char.to_int(C).
:- func hash_of(char) = int.
hash_of(C) = char.hash(C).
:- func lower(char) = char.
lower(C) = char.to_lower(C).
:- func upper(char) = char.
upper(C) = char.to_upper(C).
