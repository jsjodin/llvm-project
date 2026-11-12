% Self-checking tests of the wire format for messages to other nodes
% (runtime/encode.cpp), for which mmc cannot be the oracle. The program
% reaches the encoder through foreign_procs of its own, as nothing in the
% library uses it yet, with the type_infos code generation makes, and passes
% encodings as hex strings, which it can cut and change.
%
% - Golden bytes: small values' encodings, which a device's encoder must
%   give too.
% - Round trips: pseudo-random terms of a type with every kind of value a
%   message can hold, decoded equal to the originals, by unification and
%   by compare, in monomorphic and polymorphic code.
% - Hostile input: every truncation of encodings, an extra item, and
%   pseudo-random changes of bytes, which must be rejected or decode to a
%   value that can be written, without a crash; and lengths, numbers and
%   ranges out of bounds, a NUL in a string, padding that is not zeros, and
%   nesting past the limit, each rejected with its reason.
% - Arrays of flat records: golden bytes, as C lays the records out, round
%   trips, and records whose padding, enumeration or character is not one
%   a record of the type has, rejected with the reason, as is a length
%   past the bytes.
% - Fingerprints: equal for the same type in monomorphic and polymorphic
%   code, different for different types, and golden for a few.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:1: 0100000000000000
% OUTPUT-NEXT:-1i8: ffffffffffffffff
% OUTPUT-NEXT:255u8: ff00000000000000
% OUTPUT-NEXT:1.5: 000000000000f83f
% OUTPUT-NEXT:'é': e900000000000000
% OUTPUT-NEXT:"ab": 02000000000000006162000000000000
% OUTPUT-NEXT:"": 0000000000000000
% OUTPUT-NEXT:[1, 2]: 01000000000000000100000000000000010000000000000002000000000000000000000000000000
% OUTPUT-NEXT:array([1.5]): 0100000000000000000000000000f83f
% OUTPUT-NEXT:array([1u8, 255u8, 7u8]): 030000000000000001ff070000000000
% OUTPUT-NEXT:array([-2i16, 3i16]): 0200000000000000feff030000000000
% OUTPUT-NEXT:array(['a', 'é']): 020000000000000061000000e9000000
% OUTPUT-NEXT:array([blue, red]): 02000000000000000200000000000000
% OUTPUT-NEXT:{1, "a"}: 0000000000000000010000000000000001000000000000006100000000000000
% OUTPUT-NEXT:blue: 0200000000000000
% OUTPUT-NEXT:yes(unit): 01000000000000000000000000000000
% OUTPUT-NEXT:temp(celsius(1.5)): 0d000000000000000000000000000000000000000000f83f
% OUTPUT-NEXT:round trips: 3000 checked, 0 differ
% OUTPUT-NEXT:round trips through a type_info: 3000 checked, 0 differ
% OUTPUT-NEXT:pairs of maybe lists and tuples: 3000 checked, 0 differ
% OUTPUT-NEXT:a list of 100000 terms: 1 checked, 0 differ
% OUTPUT-NEXT:truncations: 0 of 9168 decoded
% OUTPUT-NEXT:an extra item: 0 of 200 decoded
% OUTPUT-NEXT:changed bytes: 20000 decoded or rejected, none crashed
% OUTPUT-NEXT:closure: a closure cannot be sent to another node yet
% OUTPUT-NEXT:huge string: a string is longer than the bytes left
% OUTPUT-NEXT:huge array: an array is longer than the bytes left
% OUTPUT-NEXT:functor number: a functor's number is not one of its type's
% OUTPUT-NEXT:enum number: a functor's number is not one of its type's
% OUTPUT-NEXT:int8 range: an integer is out of its type's range
% OUTPUT-NEXT:uint16 range: an integer is out of its type's range
% OUTPUT-NEXT:char range: a character is not a code point
% OUTPUT-NEXT:narrow padding: an array's padding is not zeros
% OUTPUT-NEXT:narrow length: an array is longer than the bytes left
% OUTPUT-NEXT:narrow char range: a character is not a code point
% OUTPUT-NEXT:narrow enum number: a functor's number is not one of its type's
% OUTPUT-NEXT:NUL in a string: a string has a NUL
% OUTPUT-NEXT:padding: a string's padding is not zeros
% OUTPUT-NEXT:empty: the bytes end within a value
% OUTPUT-NEXT:nested 9000 deep: decoded, equal
% OUTPUT-NEXT:nested 10001 deep: the value nests too deeply
% OUTPUT-NEXT:a list of 50000 decoded from bytes: yes
% OUTPUT-NEXT:array([rec(-1i8, blue, 'a', point(3i16, -2i16), 1.5)]): 0100000000000000ff00000002000000610000000300feff000000000000f83f
% OUTPUT-NEXT:array([{1i8, 2.5}]): 010000000000000001000000000000000000000000000440
% OUTPUT-NEXT:arrays of records: 61 checked, 0 differ
% OUTPUT-NEXT:record padding: a record's padding is not zeros
% OUTPUT-NEXT:record tail padding: a record's padding is not zeros
% OUTPUT-NEXT:record enum number: a functor's number is not one of its type's
% OUTPUT-NEXT:record char range: a character is not a code point
% OUTPUT-NEXT:record length: an array is longer than the bytes left
% OUTPUT-NEXT:fingerprint of t in polymorphic code is the same: yes
% OUTPUT-NEXT:fingerprints of list(int) and list(string) differ: yes
% OUTPUT-NEXT:fingerprints of list(int) and list(int8) differ: yes
% OUTPUT-NEXT:fingerprints of t and wrapped differ: yes
% OUTPUT-NEXT:fingerprints of a(int) and b(int) differ: yes
% OUTPUT-NEXT:fingerprints of maybe(t) and maybe(wrapped) differ: yes
% OUTPUT-NEXT:fingerprint of int: 2dbe6495aa04aa149d821b10f6cb4e45
% OUTPUT-NEXT:fingerprint of list(int): 9ed824ac14d9485424f301d89c6cffbd

:- module encode_checks.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, bool, char, float, int, int8, int16, int32, int64,
    list, maybe, pair, string, uint, uint8, uint16, uint32, uint64.

%---------------------------------------------------------------------------%
% The encoder, through hex strings.

:- pragma foreign_decl("C", local, "
#include <stdlib.h>
#include <string.h>
extern const char *merk_encode(MR_Word type_info, MR_Word value,
    unsigned char **bytes, int64_t *size);
extern const char *merk_decode(MR_Word type_info, const unsigned char *bytes,
    int64_t size, MR_Word *value);
extern void merk_fingerprint(MR_Word type_info, unsigned char *fingerprint);
extern char *merk_string_new(int64_t bytes);

static MR_String hex_of(const unsigned char *bytes, int64_t size) {
    static const char digits[] = \"0123456789abcdef\";
    MR_String hex = merk_string_new(2 * size);
    for (int64_t i = 0; i < size; ++i) {
        hex[2 * i] = digits[bytes[i] >> 4];
        hex[2 * i + 1] = digits[bytes[i] & 15];
    }
    return hex;
}

static int digit_of(char c) {
    return c <= '9' ? c - '0' : c - 'a' + 10;
}
").

    % The encoding of a value, or the reason it has none after `error: '.
:- func encode(T) = string.
:- pragma foreign_proc("C",
    encode(X::in) = (Hex::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    unsigned char *bytes;
    int64_t size;
    const char *error = merk_encode(TypeInfo_for_T, X, &bytes, &size);
    if (error) {
        Hex = merk_string_new(7 + strlen(error));
        strcpy(Hex, \"error: \");
        strcat(Hex, error);
    } else {
        Hex = hex_of(bytes, size);
        free(bytes);
    }
").

    % A value from an encoding, its hex digits in pairs; an error is the
    % reason the bytes are not one, or """" if they are.
:- pred decode(string::in, T::out, string::out) is det.
:- pragma foreign_proc("C",
    decode(Hex::in, X::out, Error::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    int64_t size = strlen(Hex) / 2;
    unsigned char *bytes = malloc(size + 1);
    for (int64_t i = 0; i < size; ++i) {
        bytes[i] = (digit_of(Hex[2 * i]) << 4) | digit_of(Hex[2 * i + 1]);
    }
    X = 0;
    const char *error = merk_decode(TypeInfo_for_T, bytes, size, &X);
    free(bytes);
    Error = merk_string_new(error ? strlen(error) : 0);
    if (error) {
        strcpy(Error, error);
    }
").

    % The fingerprint of the value's type.
:- func fingerprint(T) = string.
:- pragma foreign_proc("C",
    fingerprint(_X::in) = (Hex::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    unsigned char fingerprint[16];
    merk_fingerprint(TypeInfo_for_T, fingerprint);
    Hex = hex_of(fingerprint, 16);
").

    % Decodes an encoding as a value of the example's type.
:- pred decode_as(T::in, string::in, T::out, string::out) is det.
decode_as(_, Hex, X, Error) :-
    decode(Hex, X, Error).

%---------------------------------------------------------------------------%
% Pseudo-random numbers.

:- type seed == int.

:- pred next(int::in, int::out, seed::in, seed::out) is det.
next(Bound, N, S0, S) :-
    S = (S0 * 1103515245 + 12345) /\ 2147483647,
    N = (S >> 8) mod Bound.

%---------------------------------------------------------------------------%
% Terms with every kind of value a message can hold: integers of each size,
% floats, strings, an ill-formed one too, characters, an enum, a dummy, a
% notag of a float and one of a term, tuples of one and of two, lists,
% arrays, narrow ones too, polymorphic types, and more constructors with arguments than
% there are tags, so that the last tag is shared and those cells have
% secondary tags.

:- type t
    --->    leaf
    ;       num(int)
    ;       real(float)
    ;       text(string)
    ;       ch(char)
    ;       pr(t, t)
    ;       many(list(t))
    ;       box(wrapped)
    ;       tup({int, t})
    ;       arr(array(t))
    ;       opt(maybe(t))
    ;       hue(colour)
    ;       small(int8)
    ;       temp(celsius)
    ;       wide(int16, int32, int64, uint, uint8, uint16, uint32, uint64)
    ;       one({string})
    ;       narrow(array(uint8), array(int16), array(char), array(colour))
    ;       nothing(unit).

:- type wrapped ---> wrapped(t).
:- type celsius ---> celsius(float).
:- type colour ---> red ; green ; blue.
:- type unit ---> unit.
:- type a ---> a(int).
:- type b ---> a(int).

:- pred gen(int::in, t::out, seed::in, seed::out) is det.
gen(Depth, T, !S) :-
    next(( if Depth =< 0 then 7 else 18 ), K, !S),
    ( if K = 0 then
        T = leaf
    else if K = 1 then
        next(2000001, N, !S),
        T = num(N - 1000000)
    else if K = 2 then
        next(1000, N, !S),
        T = real(float(N - 500) / 8.0)
    else if K = 3 then
        next(7, N, !S),
        T = text(det_index0(["", "a", "λx", "two words", "\"quoted\"",
            "日本語 sixteen!!", string.between("é", 0, 1)], N))
    else if K = 4 then
        next(5, N, !S),
        T = ch(det_index0(['a', 'Z', 'λ', '\n', '😀'], N))
    else if K = 5 then
        next(3, N, !S),
        T = hue(det_index0([red, green, blue], N))
    else if K = 6 then
        next(3, N, !S),
        T = small(det_index0([-128i8, 0i8, 127i8], N))
    else if K = 7 then
        gen(Depth - 1, A, !S),
        gen(Depth - 1, B, !S),
        T = pr(A, B)
    else if K = 8 then
        next(5, N, !S),
        gen_list(N, Depth - 1, L, !S),
        T = many(L)
    else if K = 9 then
        gen(Depth - 1, A, !S),
        T = box(wrapped(A))
    else if K = 10 then
        next(100, N, !S),
        gen(Depth - 1, A, !S),
        T = tup({N, A})
    else if K = 11 then
        next(4, N, !S),
        gen_list(N, Depth - 1, L, !S),
        T = arr(array(L))
    else if K = 12 then
        next(2, N, !S),
        gen(Depth - 1, A, !S),
        T = opt(( if N = 0 then no else yes(A) ))
    else if K = 13 then
        next(400, N, !S),
        T = temp(celsius(float(N) / 4.0))
    else if K = 14 then
        next(2, N, !S),
        ( if N = 0 then
            T = wide(-32768i16, -2147483648i32, -9223372036854775808i64,
                18446744073709551615u, 255u8, 65535u16, 4294967295u32,
                18446744073709551615u64)
        else
            T = wide(32767i16, 2147483647i32, 9223372036854775807i64,
                0u, 0u8, 0u16, 0u32, 0u64)
        )
    else if K = 15 then
        T = one({"single"})
    else if K = 16 then
        next(20, N, !S),
        T = narrow(
            array.generate(N, (func(I) = uint8.cast_from_int(I * 41 + N))),
            array.generate(N mod 5, (func(I) = int16.cast_from_int(I - 2))),
            array.generate(N mod 3, (func(I) = det_index0(['a', 'λ', '😀'], I))),
            array.generate(N mod 4, (func(I) = det_index0([red, green, blue,
                red], I))))
    else
        T = nothing(unit)
    ).

:- pred gen_list(int::in, int::in, list(t)::out, seed::in, seed::out) is det.
gen_list(N, Depth, L, !S) :-
    ( if N =< 0 then
        L = []
    else
        gen(Depth, H, !S),
        gen_list(N - 1, Depth, T, !S),
        L = [H | T]
    ).

%---------------------------------------------------------------------------%
% Golden bytes.

:- pred golden(string::in, T::in, io::di, io::uo) is det.
golden(What, X, !IO) :-
    io.write_string(What ++ ": " ++ encode(X) ++ "\n", !IO).

:- pred check_golden(io::di, io::uo) is det.
check_golden(!IO) :-
    golden("1", 1, !IO),
    golden("-1i8", -1i8, !IO),
    golden("255u8", 255u8, !IO),
    golden("1.5", 1.5, !IO),
    golden("'é'", 'é', !IO),
    golden("\"ab\"", "ab", !IO),
    golden("\"\"", "", !IO),
    golden("[1, 2]", [1, 2], !IO),
    golden("array([1.5])", array([1.5]), !IO),
    golden("array([1u8, 255u8, 7u8])", array([1u8, 255u8, 7u8]), !IO),
    golden("array([-2i16, 3i16])", array([-2i16, 3i16]), !IO),
    golden("array(['a', 'é'])", array(['a', 'é']), !IO),
    golden("array([blue, red])", array([blue, red]), !IO),
    golden("{1, \"a\"}", {1, "a"}, !IO),
    golden("blue", blue, !IO),
    golden("yes(unit)", yes(unit), !IO),
    golden("temp(celsius(1.5))", temp(celsius(1.5)), !IO).

%---------------------------------------------------------------------------%
% Round trips.

    % The terms that do not come back equal, by unification and by compare,
    % in code polymorphic in their type.
:- pred count_bad(list(T)::in, int::in, int::out) is det.
count_bad([], !Bad).
count_bad([X | Xs], !Bad) :-
    decode(encode(X), Y, Error),
    ( if Error = "", X = Y, compare((=), X, Y) then
        true
    else
        !:Bad = !.Bad + 1
    ),
    count_bad(Xs, !Bad).

:- pred report(string::in, int::in, int::in, io::di, io::uo) is det.
report(What, Count, Bad, !IO) :-
    io.write_string(What ++ ": " ++ int_to_string(Count) ++ " checked, " ++
        int_to_string(Bad) ++ " differ\n", !IO).

:- pred check_round_trips(list(t)::in, io::di, io::uo) is det.
check_round_trips(Terms, !IO) :-
    count_mono(Terms, 0, Bad1),
    report("round trips", length(Terms), Bad1, !IO),
    count_bad(Terms, 0, Bad2),
    report("round trips through a type_info", length(Terms), Bad2, !IO),
    Wrapped = list.map((func(T) = yes([T, leaf]) - {T, "x"}), Terms),
    count_bad(Wrapped, 0, Bad3),
    report("pairs of maybe lists and tuples", length(Wrapped), Bad3, !IO),
    gen_list(100000, 2, Long, 777, _),
    count_bad([Long], 0, Bad4),
    report("a list of 100000 terms", 1, Bad4, !IO).

    % As count_bad, at t.
:- pred count_mono(list(t)::in, int::in, int::out) is det.
count_mono([], !Bad).
count_mono([X | Xs], !Bad) :-
    decode(encode(X), Y : t, Error),
    ( if Error = "", X = Y, compare((=), X, Y) then
        true
    else
        !:Bad = !.Bad + 1
    ),
    count_mono(Xs, !Bad).

%---------------------------------------------------------------------------%
% Hostile input.

    % How many of the encoding's proper prefixes, at item boundaries and
    % between them, decode.
:- pred truncations(t::in, int::in, int::out, int::in, int::out) is det.
truncations(X, !Tried, !Decoded) :-
    Hex = encode(X),
    truncate_from(X, Hex, 0, length(Hex) / 2, !Tried, !Decoded).

:- pred truncate_from(t::in, string::in, int::in, int::in,
    int::in, int::out, int::in, int::out) is det.
truncate_from(X, Hex, Bytes, Size, !Tried, !Decoded) :-
    ( if Bytes >= Size then
        true
    else
        decode_as(X, left(Hex, 2 * Bytes), _, Error),
        !:Tried = !.Tried + 1,
        ( if Error = "" then !:Decoded = !.Decoded + 1 else true ),
        truncate_from(X, Hex, Bytes + 1, Size, !Tried, !Decoded)
    ).

:- pred extra_items(list(t)::in, int::in, int::out) is det.
extra_items([], !Decoded).
extra_items([X | Xs], !Decoded) :-
    decode_as(X, encode(X) ++ "0000000000000000", _, Error),
    ( if Error = "" then !:Decoded = !.Decoded + 1 else true ),
    extra_items(Xs, !Decoded).

    % Changes a byte of a term's encoding to a pseudo-random one, or two,
    % and decodes it; what decodes is written, which walks all of it.
:- pred change_bytes(int::in, list(t)::in, int::in, int::out,
    seed::in, seed::out) is det.
change_bytes(N, Terms, !Written, !S) :-
    ( if N =< 0 then
        true
    else
        next(list.length(Terms), I, !S),
        X = det_index0(Terms, I) : t,
        Hex0 = encode(X),
        change_byte(Hex0, Hex1, !S),
        next(2, Twice, !S),
        ( if Twice = 0 then Hex = Hex1 else change_byte(Hex1, Hex, !S) ),
        decode_as(X, Hex, Y, Error),
        ( if Error = "" then
            !:Written = !.Written + length(string.string(Y))
        else
            true
        ),
        change_bytes(N - 1, Terms, !Written, !S)
    ).

:- pred change_byte(string::in, string::out, seed::in, seed::out) is det.
change_byte(Hex0, Hex, !S) :-
    Size = length(Hex0) / 2,
    ( if Size = 0 then
        Hex = Hex0
    else
        next(Size, At, !S),
        next(256, Byte, !S),
        Hex = left(Hex0, 2 * At) ++ byte_hex(Byte) ++
            right(Hex0, length(Hex0) - 2 * At - 2)
    ).

:- func byte_hex(int) = string.
byte_hex(Byte) = string.format("%02x", [i(Byte)]).

    % An item's hex digits, little-endian.
:- func item(int) = string.
item(N) = string.format("%02x%02x%02x%02x%02x%02x%02x%02x",
    [i(N /\ 255), i((N >> 8) /\ 255), i((N >> 16) /\ 255),
    i((N >> 24) /\ 255), i((N >> 32) /\ 255), i((N >> 40) /\ 255),
    i((N >> 48) /\ 255), i((N >> 56) /\ 255)]).

:- pred rejected(string::in, T::in, string::in, io::di, io::uo) is det.
rejected(What, Example, Hex, !IO) :-
    decode_as(Example, Hex, _, Error),
    io.write_string(What ++ ": " ++
        ( if Error = "" then "decoded" else Error ) ++ "\n", !IO).

    % pr(pr(... pr(leaf, leaf) ...), leaf), nested through pr's first
    % argument, which is not a loop as its second is.
:- func nested(int) = t.
nested(N) = ( if N =< 0 then leaf else pr(nested(N - 1), leaf) ).

:- func nested_hex(int) = string.
nested_hex(N) = duplicate(N, item(5)) ++ item(0) ++ duplicate(N, item(0)).

:- func duplicate(int, string) = string.
duplicate(N, S) = append_list(list.duplicate(N, S)).

:- pred check_hostile(list(t)::in, io::di, io::uo) is det.
check_hostile(Terms, !IO) :-
    Few = take_upto(200, Terms),
    list.foldl2(truncations, Few, 0, Tried, 0, Decoded),
    io.write_string("truncations: " ++ int_to_string(Decoded) ++ " of " ++
        int_to_string(Tried) ++ " decoded\n", !IO),
    extra_items(Few, 0, Extra),
    io.write_string("an extra item: " ++ int_to_string(Extra) ++ " of " ++
        int_to_string(length(Few)) ++ " decoded\n", !IO),
    change_bytes(20000, Few, 0, _, 99, _),
    io.write_string("changed bytes: 20000 decoded or rejected, " ++
        "none crashed\n", !IO),
    Closure = (func(X) = X + 1) : (func(int) = int),
    io.write_string("closure: " ++
        remove_error_prefix(encode(Closure)) ++ "\n", !IO),
    rejected("huge string", "", item(1 << 40) ++ item(0), !IO),
    rejected("huge array", array([1]), item(1 << 40) ++ item(0), !IO),
    rejected("functor number", leaf, item(18), !IO),
    rejected("enum number", red, item(3), !IO),
    rejected("int8 range", 0i8, item(128), !IO),
    rejected("uint16 range", 0u16, item(65536), !IO),
    rejected("char range", 'a', item(0x110000), !IO),
    rejected("narrow padding", array([1u8]), item(1) ++ "0100000000000009",
        !IO),
    rejected("narrow length", array([1u8]), item(9) ++ item(0), !IO),
    rejected("narrow char range", array(['a']),
        item(1) ++ "0000110000000000", !IO),
    rejected("narrow enum number", array([red]),
        item(1) ++ "0300000000000000", !IO),
    rejected("NUL in a string", "", item(2) ++ "6100000000000000", !IO),
    rejected("padding", "", item(1) ++ "6101000000000000", !IO),
    rejected("empty", 0, "", !IO),
    decode_as(leaf, nested_hex(9000), Deep, DeepError),
    io.write_string("nested 9000 deep: " ++
        ( if DeepError = "" then
            "decoded, " ++ ( if Deep = nested(9000) then "equal" else "not equal" )
        else
            DeepError
        ) ++ "\n", !IO),
    rejected("nested 10001 deep", leaf, nested_hex(10001), !IO),
    % A list's tail is a loop, so its length has no limit.
    ListHex = item(1) ++ list_items(50000) ++ item(0),
    decode_as([0], ListHex, List, ListError),
    io.write_string("a list of 50000 decoded from bytes: " ++
        ( if ListError = "", List = 1 .. 50000 then "yes" else "no" ) ++
        "\n", !IO).

    % The items of 1 .. N as a list's, but its first functor number.
:- func list_items(int) = string.
list_items(N) = join_list(item(1), list.map(item, 1 .. N)).

:- func remove_error_prefix(string) = string.
remove_error_prefix(S) =
    ( if remove_prefix("error: ", S, Rest) then Rest else S ).

%---------------------------------------------------------------------------%
% Fingerprints.

:- func poly_fingerprint(T) = string.
poly_fingerprint(X) = fingerprint(X).

:- pred differ(string::in, string::in, string::in, io::di, io::uo) is det.
differ(What, F1, F2, !IO) :-
    io.write_string("fingerprints of " ++ What ++ " differ: " ++
        ( if F1 = F2 then "no" else "yes" ) ++ "\n", !IO).

:- pred check_fingerprints(io::di, io::uo) is det.
check_fingerprints(!IO) :-
    io.write_string("fingerprint of t in polymorphic code is the same: " ++
        ( if fingerprint(leaf) = poly_fingerprint(leaf) then "yes" else "no" )
        ++ "\n", !IO),
    differ("list(int) and list(string)", fingerprint([1]),
        fingerprint(["a"]), !IO),
    differ("list(int) and list(int8)", fingerprint([1]), fingerprint([1i8]),
        !IO),
    differ("t and wrapped", fingerprint(leaf), fingerprint(wrapped(leaf)),
        !IO),
    differ("a(int) and b(int)", fingerprint(a(1) : a), fingerprint(a(1) : b),
        !IO),
    differ("maybe(t) and maybe(wrapped)", fingerprint(yes(leaf)),
        fingerprint(yes(wrapped(leaf))), !IO),
    io.write_string("fingerprint of int: " ++ fingerprint(1) ++ "\n", !IO),
    io.write_string("fingerprint of list(int): " ++ fingerprint([1]) ++ "\n",
        !IO).

%---------------------------------------------------------------------------%
% Arrays of flat records, whose records travel as their bytes, as C lays
% them out (runtime/record.h): a rec is an int8 at 0, a colour at 4, a char
% at 8, a point of two int16s at 12 and a float at 16, 24 bytes.

:- type point ---> point(int16, int16).
:- type rec
    --->    rec(
                small :: int8,
                hue :: colour,
                letter :: char,
                at :: point,
                x :: float
            ).

:- func rec(int) = rec.
rec(I) = rec(int8.cast_from_int(I * 37 - 128), det_index0([red, green, blue],
    I mod 3), det_index0(['a', 'λ', '😀', 'Z'], I mod 4),
    point(int16.cast_from_int(I * 1000), int16.cast_from_int(-I)),
    float(I) / 8.0).

:- func recs(int) = array(rec).
recs(N) = array.generate(N, rec).

    % The arrays that do not come back equal, by unification and compare.
:- pred count_bad_recs(int::in, int::in, int::in, int::out) is det.
count_bad_recs(N, Max, !Bad) :-
    ( if N =< Max then
        A = recs(N),
        decode_as(A, encode(A), B, Error),
        ( if Error = "", A = B, compare((=), A, B) then
            true
        else
            !:Bad = !.Bad + 1
        ),
        count_bad_recs(N + 1, Max, !Bad)
    else
        true
    ).

:- pred check_records(io::di, io::uo) is det.
check_records(!IO) :-
    golden("array([rec(-1i8, blue, 'a', point(3i16, -2i16), 1.5)])",
        array([rec(-1i8, blue, 'a', point(3i16, -2i16), 1.5)]), !IO),
    golden("array([{1i8, 2.5}])", array([{1i8, 2.5}]), !IO),
    count_bad_recs(0, 60, 0, Bad),
    Arrays = list.map(recs, 0 .. 60),
    count_bad(Arrays, 0, PolyBad),
    io.write_string("arrays of records: 61 checked, " ++
        int_to_string(Bad + PolyBad) ++ " differ\n", !IO),
    One = item(1),
    Fields = "02000000" ++ "61000000" ++ "0300feff" ++ "000000000000f83f",
    rejected("record padding", recs(1), One ++ "ff010000" ++ Fields, !IO),
    rejected("record tail padding", array([{1i8, 2.5}]),
        One ++ "0100000000000001" ++ "0000000000000440", !IO),
    rejected("record enum number", recs(1),
        One ++ "ff000000" ++ "03000000" ++ "61000000" ++ "0300feff" ++
        "000000000000f83f", !IO),
    rejected("record char range", recs(1),
        One ++ "ff000000" ++ "02000000" ++ "00001100" ++ "0300feff" ++
        "000000000000f83f", !IO),
    rejected("record length", recs(1),
        item(2) ++ "ff000000" ++ Fields, !IO).

%---------------------------------------------------------------------------%

main(!IO) :-
    check_golden(!IO),
    gen_list(3000, 5, Terms, 12345, _),
    check_round_trips(Terms, !IO),
    check_hostile(Terms, !IO),
    check_records(!IO),
    check_fingerprints(!IO).
