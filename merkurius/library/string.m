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

:- implementation.
:- import_module int.
:- import_module require.

% Strings are UTF-8 ending in a NUL, as in mmc's C grades, and indices
% count bytes. A byte that is not part of a well-formed sequence reads as
% U+FFFD, one for each such byte. What works on bytes is in C, in
% runtime/string.cpp; the rest is written in Mercury on it. The behaviour
% where the manual leaves it open is mmc's, found by running its programs.

:- pragma foreign_decl("C", local, "
extern MR_Integer merk_char_bytes(MR_Char code);
extern MR_String merk_string_new(MR_Integer bytes);
extern MR_Integer merk_string_put(MR_String text, MR_Integer offset,
    MR_Char code);
extern MR_Integer merk_string_put_string(MR_String text, MR_Integer offset,
    MR_ConstString from);
extern MR_Char merk_string_decode(MR_ConstString text, MR_Integer offset,
    MR_Integer *next);
extern MR_Integer merk_string_previous(MR_ConstString text,
    MR_Integer offset);
extern MR_Integer merk_string_count_codepoints(MR_ConstString text);
extern MR_String merk_string_append(MR_ConstString a, MR_ConstString b);
extern MR_String merk_string_between(MR_ConstString text, MR_Integer start,
    MR_Integer end);
extern MR_Integer merk_string_search(MR_ConstString text,
    MR_ConstString pattern);
extern MR_Integer merk_string_at(MR_ConstString text, MR_Integer offset,
    MR_ConstString pattern);
extern MR_String merk_string_case(MR_ConstString text, MR_Integer upper);
extern MR_String merk_string_strip(MR_ConstString text, MR_Integer leading,
    MR_Integer trailing);
extern MR_Integer merk_char_is_whitespace(MR_Char code);
extern MR_String merk_string_replace_all(MR_ConstString text,
    MR_ConstString pattern, MR_ConstString replacement);
extern MR_Integer merk_string_to_int(MR_ConstString text, MR_Integer *value);
extern MR_Integer merk_string_to_float(MR_ConstString text, double *value);
extern MR_String merk_string_from_int(MR_Integer value);
extern MR_String merk_string_from_float(MR_Float value);
#include <string.h>
").

%---------------------------------------------------------------------------%
% The primitives.

    % The number of bytes.
:- func bytes(string) = int.
:- pragma foreign_proc("C",
    bytes(S::in) = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = (MR_Integer) strlen(S);
").

    % The code point at a byte, which must be before the NUL, and the byte
    % after it.
:- pred decode(string::in, int::in, char::out, int::out) is det.
:- pragma foreign_proc("C",
    decode(S::in, Offset::in, C::out, Next::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    C = merk_string_decode(S, Offset, &Next);
").

    % Where the code point that ends at a byte starts.
:- func previous(string, int) = int.
:- pragma foreign_proc("C",
    previous(S::in, Offset::in) = (Start::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Start = merk_string_previous(S, Offset);
").

    % The bytes of a code point's UTF-8, or 0 for one that no string may
    % hold: NUL or a surrogate.
:- func char_bytes(char) = int.
:- pragma foreign_proc("C",
    char_bytes(C::in) = (N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = merk_char_bytes(C);
").

    % A string of so many bytes, to be filled in.
:- pred new_string(int::in, string::uo) is det.
:- pragma foreign_proc("C",
    new_string(Bytes::in, S::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    S = merk_string_new(Bytes);
").

    % Writes a code point at a byte, and gives the byte after it.
:- pred put_char(char::in, int::in, int::out, string::di, string::uo) is det.
:- pragma foreign_proc("C",
    put_char(C::in, Offset::in, Next::out, S0::di, S::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Next = merk_string_put(S0, Offset, C);
    S = S0;
").

    % Writes a string's bytes at a byte, and gives the byte after them.
:- pred put_string(string::in, int::in, int::out, string::di, string::uo)
    is det.
:- pragma foreign_proc("C",
    put_string(From::in, Offset::in, Next::out, S0::di, S::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Next = merk_string_put_string(S0, Offset, From);
    S = S0;
").

    % The bytes from Start to just before End, each clamped to the string.
:- func bytes_between(string, int, int) = string.
:- mode bytes_between(in, in, in) = uo is det.
:- pragma foreign_proc("C",
    bytes_between(S::in, Start::in, End::in) = (R::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    R = merk_string_between(S, Start, End);
").

    % Whether a string's bytes are at a byte of another.
:- pred at(string::in, int::in, string::in) is semidet.
:- pragma foreign_proc("C",
    at(S::in, Offset::in, Pattern::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = merk_string_at(S, Offset, Pattern);
").

:- pred same(string::in, string::in) is semidet.
:- pragma foreign_proc("C",
    same(A::in, B::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = strcmp(A, B) == 0;
").

:- pred is_space(char::in) is semidet.
:- pragma foreign_proc("C",
    is_space(C::in),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    SUCCESS_INDICATOR = merk_char_is_whitespace(C);
").

    % The error mmc gives for a list that no string can hold.
:- pred bad_char_list is erroneous.

bad_char_list :-
    unexpected("predicate `string.from_char_list'/2",
        "null character or surrogate code point in list").

%---------------------------------------------------------------------------%
% Conversions between strings and lists of characters.

to_char_list(S) = Cs :-
    to_char_list(S, Cs).

:- pragma promise_equivalent_clauses(pred(to_char_list/2)).
to_char_list(S::in, Cs::out) :-
    chars_before(S, bytes(S), [], Cs).
to_char_list(S::uo, Cs::in) :-
    from_char_list(Cs, S).

    % The code points that end at Offset, before Cs0.
:- pred chars_before(string::in, int::in, list(char)::in, list(char)::out)
    is det.

chars_before(S, Offset, Cs0, Cs) :-
    ( if Offset =< 0 then
        Cs = Cs0
    else
        Start = previous(S, Offset),
        decode(S, Start, C, Next),
        ( if Next = Offset then
            chars_before(S, Start, [C | Cs0], Cs)
        else
            % Start is a lead byte whose sequence is ill-formed: the byte
            % before Offset is one U+FFFD.
            chars_before(S, Offset - 1, [replacement | Cs0], Cs)
        )
    ).

:- func replacement = char.

replacement = '�'.

to_rev_char_list(S) = Cs :-
    to_rev_char_list(S, Cs).

:- pragma promise_equivalent_clauses(pred(to_rev_char_list/2)).
to_rev_char_list(S::in, Cs::out) :-
    rev_chars_from(S, 0, bytes(S), [], Cs).
to_rev_char_list(S::uo, Cs::in) :-
    from_rev_char_list(Cs, S).

:- pred rev_chars_from(string::in, int::in, int::in, list(char)::in,
    list(char)::out) is det.

rev_chars_from(S, Offset, End, Cs0, Cs) :-
    ( if Offset >= End then
        Cs = Cs0
    else
        decode(S, Offset, C, Next),
        rev_chars_from(S, Next, End, [C | Cs0], Cs)
    ).

from_char_list(Cs) = S :-
    from_char_list(Cs, S).

:- pragma promise_equivalent_clauses(pred(from_char_list/2)).
from_char_list(Cs::in, S::uo) :-
    new_string(list_bytes(Cs, 0), S0),
    put_chars(Cs, 0, S0, S).
from_char_list(Cs::out, S::in) :-
    to_char_list(S, Cs).

    % The bytes of the code points' UTF-8.
:- func list_bytes(list(char), int) = int.

list_bytes([], N) = N.
list_bytes([C | Cs], N0) = N :-
    Bytes = char_bytes(C),
    ( if Bytes = 0 then
        bad_char_list
    else
        N = list_bytes(Cs, N0 + Bytes)
    ).

:- pred put_chars(list(char)::in, int::in, string::di, string::uo) is det.

put_chars([], _, !S).
put_chars([C | Cs], Offset, !S) :-
    put_char(C, Offset, Next, !S),
    put_chars(Cs, Next, !S).

:- pred from_rev_char_list(list(char)::in, string::uo) is det.

from_rev_char_list(Cs, S) :-
    Bytes = list_bytes(Cs, 0),
    new_string(Bytes, S0),
    put_rev_chars(Cs, Bytes, S0, S).

    % The code points, the last first, ending at End.
:- pred put_rev_chars(list(char)::in, int::in, string::di, string::uo) is det.

put_rev_chars([], _, !S).
put_rev_chars([C | Cs], End, !S) :-
    Start = End - char_bytes(C),
    put_char(C, Start, _, !S),
    put_rev_chars(Cs, Start, !S).

duplicate_char(C, N) = S :-
    duplicate_char(C, N, S).

duplicate_char(C, N, S) :-
    Bytes = char_bytes(C),
    ( if Bytes = 0 then
        bad_char_list
    else
        Count = max(N, 0),
        new_string(Count * Bytes, S0),
        put_copies(C, Count, 0, S0, S)
    ).

:- pred put_copies(char::in, int::in, int::in, string::di, string::uo)
    is det.

put_copies(C, N, Offset, !S) :-
    ( if N =< 0 then
        true
    else
        put_char(C, Offset, Next, !S),
        put_copies(C, N - 1, Next, !S)
    ).

%---------------------------------------------------------------------------%
% Reading characters.

index(S, I, C) :-
    I >= 0,
    I < bytes(S),
    decode(S, I, C0, _),
    C = unsafe_promise_unique_char(C0).

:- func unsafe_promise_unique_char(char) = char.
:- mode unsafe_promise_unique_char(in) = uo is det.
:- pragma foreign_proc("C",
    unsafe_promise_unique_char(C0::in) = (C::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    C = C0;
").

det_index(S, I) = C :-
    det_index(S, I, C).

det_index(S, I, C) :-
    ( if index(S, I, C0) then
        C = C0
    else
        unexpected("predicate `string.det_index'/3", "index out of range")
    ).

%---------------------------------------------------------------------------%
% Lengths.

length(S) = N :-
    length(S, N).

:- pragma foreign_proc("C",
    length(S::in, N::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = (MR_Integer) strlen(S);
").
:- pragma foreign_proc("C",
    length(S::ui, N::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = (MR_Integer) strlen(S);
").

count_codepoints(S) = N :-
    count_codepoints(S, N).

:- pragma foreign_proc("C",
    count_codepoints(S::in, N::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    N = merk_string_count_codepoints(S);
").

%---------------------------------------------------------------------------%
% Tests.

is_empty(S) :-
    bytes(S) = 0.

contains_char(S, C) :-
    contains_char_from(S, C, 0, bytes(S)).

:- pred contains_char_from(string::in, char::in, int::in, int::in)
    is semidet.

contains_char_from(S, C, Offset, End) :-
    Offset < End,
    decode(S, Offset, C0, Next),
    ( C0 = C
    ; contains_char_from(S, C, Next, End)
    ).

:- pragma foreign_proc("C",
    sub_string_search(S::in, Pattern::in, I::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    I = merk_string_search(S, Pattern);
    SUCCESS_INDICATOR = I >= 0;
").

%---------------------------------------------------------------------------%
% Appending.

append(A, B) = S :-
    append(A, B, S).

:- pragma promise_equivalent_clauses(pred(append/3)).
append(A::in, B::in, S::in) :-
    bytes(A) + bytes(B) = bytes(S),
    at(S, 0, A),
    at(S, bytes(A), B).
append(A::in, B::uo, S::in) :-
    at(S, 0, A),
    B = bytes_between(S, bytes(A), bytes(S)).
append(A::in, B::in, S::uo) :-
    S = append_bytes(A, B).
append(A::uo, B::in, S::in) :-
    Start = bytes(S) - bytes(B),
    Start >= 0,
    at(S, Start, B),
    A = bytes_between(S, 0, Start).
append(A::out, B::out, S::in) :-
    split_at_boundary(S, 0, bytes(S), Offset),
    A = bytes_between(S, 0, Offset),
    B = bytes_between(S, Offset, bytes(S)).

    % The bytes that start a code point, and the end, from Offset on.
:- pred split_at_boundary(string::in, int::in, int::in, int::out) is multi.

split_at_boundary(S, Offset, End, Split) :-
    ( if Offset >= End then
        Split = End
    else
        (
            Split = Offset
        ;
            decode(S, Offset, _, Next),
            split_at_boundary(S, Next, End, Split)
        )
    ).

:- func append_bytes(string, string) = string.
:- mode append_bytes(in, in) = uo is det.
:- pragma foreign_proc("C",
    append_bytes(A::in, B::in) = (S::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    S = merk_string_append(A, B);
").

A ++ B = append_bytes(A, B).

append_list(Ss) = S :-
    append_list(Ss, S).

append_list(Ss, S) :-
    new_string(total_bytes(Ss, 0), S0),
    put_strings(Ss, 0, S0, S).

:- func total_bytes(list(string), int) = int.

total_bytes([], N) = N.
total_bytes([S | Ss], N) = total_bytes(Ss, N + bytes(S)).

:- pred put_strings(list(string)::in, int::in, string::di, string::uo)
    is det.

put_strings([], _, !S).
put_strings([From | Froms], Offset, !S) :-
    put_string(From, Offset, Next, !S),
    put_strings(Froms, Next, !S).

join_list(_, []) = S :-
    new_string(0, S).
join_list(Separator, [First | Rest]) = S :-
    Bytes = total_bytes(Rest, bytes(First)) +
        list.length(Rest) * bytes(Separator),
    new_string(Bytes, S0),
    put_string(First, 0, Next, S0, S1),
    put_separated(Separator, Rest, Next, S1, S).

:- pred put_separated(string::in, list(string)::in, int::in,
    string::di, string::uo) is det.

put_separated(_, [], _, !S).
put_separated(Separator, [From | Froms], Offset, !S) :-
    put_string(Separator, Offset, Next0, !S),
    put_string(From, Next0, Next, !S),
    put_separated(Separator, Froms, Next, !S).

%---------------------------------------------------------------------------%
% The first character.

:- pragma promise_equivalent_clauses(pred(first_char/3)).
first_char(S::in, C::in, Rest::in) :-
    first_char_parts(S, C, Rest).
first_char(S::in, C::uo, Rest::in) :-
    first_char_parts(S, C0, Rest),
    C = unsafe_promise_unique_char(C0).
first_char(S::in, C::in, Rest::uo) :-
    first_parts(S, C, Rest).
first_char(S::in, C::uo, Rest::uo) :-
    first_parts(S, C0, Rest),
    C = unsafe_promise_unique_char(C0).
first_char(S::uo, C::in, Rest::in) :-
    Bytes = char_bytes(C),
    ( if Bytes = 0 then
        bad_char_list
    else
        new_string(Bytes + bytes(Rest), S0),
        put_char(C, 0, Next, S0, S1),
        put_string(Rest, Next, _, S1, S)
    ).

:- pred first_char_parts(string::in, char::out, string::in) is semidet.

first_char_parts(S, C, Rest) :-
    bytes(S) > 0,
    decode(S, 0, C, Next),
    bytes(S) - Next = bytes(Rest),
    at(S, Next, Rest).

:- pred first_parts(string::in, char::out, string::uo) is semidet.

first_parts(S, C, Rest) :-
    bytes(S) > 0,
    decode(S, 0, C, Next),
    Rest = bytes_between(S, Next, bytes(S)).

%---------------------------------------------------------------------------%
% Substrings, in bytes.

left(S, N) = Left :-
    left(S, N, Left).

left(S, N, bytes_between(S, 0, N)).

right(S, N) = Right :-
    right(S, N, Right).

right(S, N, Right) :-
    Bytes = bytes(S),
    Right = bytes_between(S, Bytes - max(N, 0), Bytes).

between(S, Start, End) = Between :-
    between(S, Start, End, Between).

between(S, Start, End, bytes_between(S, Start, End)).

%---------------------------------------------------------------------------%
% Splitting.

words(S) = Words :-
    words_from(S, 0, bytes(S), Words).

    % The words from Offset on.
:- pred words_from(string::in, int::in, int::in, list(string)::out) is det.

words_from(S, Offset, End, Words) :-
    Start = skip_spaces(S, Offset, End),
    ( if Start >= End then
        Words = []
    else
        WordEnd = skip_word(S, Start, End),
        Words = [bytes_between(S, Start, WordEnd) | Words0],
        words_from(S, WordEnd, End, Words0)
    ).

:- func skip_spaces(string, int, int) = int.

skip_spaces(S, Offset, End) = Start :-
    ( if Offset < End, decode(S, Offset, C, Next), is_space(C) then
        Start = skip_spaces(S, Next, End)
    else
        Start = Offset
    ).

:- func skip_word(string, int, int) = int.

skip_word(S, Offset, End) = WordEnd :-
    ( if Offset < End, decode(S, Offset, C, Next), not is_space(C) then
        WordEnd = skip_word(S, Next, End)
    else
        WordEnd = Offset
    ).

split_at_char(C, S) = Parts :-
    split_from(C, S, 0, 0, bytes(S), Parts).

    % The parts from Start, the start of the current one, where Offset is
    % the next code point to look at.
:- pred split_from(char::in, string::in, int::in, int::in, int::in,
    list(string)::out) is det.

split_from(Separator, S, Start, Offset, End, Parts) :-
    ( if Offset >= End then
        Parts = [bytes_between(S, Start, End)]
    else
        decode(S, Offset, C, Next),
        ( if C = Separator then
            Parts = [bytes_between(S, Start, Offset) | Parts0],
            split_from(Separator, S, Next, Next, End, Parts0)
        else
            split_from(Separator, S, Start, Next, End, Parts)
        )
    ).

split_into_lines(S) = Lines :-
    lines_from(S, 0, bytes(S), Lines).

    % Each line ends at a newline, or at the end if it is not empty.
:- pred lines_from(string::in, int::in, int::in, list(string)::out) is det.

lines_from(S, Start, End, Lines) :-
    ( if Start >= End then
        Lines = []
    else
        LineEnd = line_end(S, Start, End),
        Lines = [bytes_between(S, Start, LineEnd) | Lines0],
        lines_from(S, LineEnd + 1, End, Lines0)
    ).

:- func line_end(string, int, int) = int.

line_end(S, Offset, End) = LineEnd :-
    ( if Offset >= End then
        LineEnd = End
    else
        decode(S, Offset, C, Next),
        ( if C = '\n' then
            LineEnd = Offset
        else
            LineEnd = line_end(S, Next, End)
        )
    ).

%---------------------------------------------------------------------------%
% Prefixes and suffixes.

:- pragma promise_equivalent_clauses(pred(prefix/2)).
prefix(S::in, Prefix::in) :-
    at(S, 0, Prefix).
prefix(S::in, Prefix::out) :-
    split_at_boundary(S, 0, bytes(S), Offset),
    Prefix = bytes_between(S, 0, Offset).

:- pragma promise_equivalent_clauses(pred(suffix/2)).
suffix(S::in, Suffix::in) :-
    Start = bytes(S) - bytes(Suffix),
    Start >= 0,
    at(S, Start, Suffix).
suffix(S::in, Suffix::out) :-
    split_at_boundary(S, 0, bytes(S), Offset),
    Suffix = bytes_between(S, Offset, bytes(S)).

remove_prefix(Prefix, S, Rest) :-
    at(S, 0, Prefix),
    Rest = bytes_between(S, bytes(Prefix), bytes(S)).

remove_suffix(S, Suffix, Rest) :-
    Start = bytes(S) - bytes(Suffix),
    Start >= 0,
    at(S, Start, Suffix),
    Rest = bytes_between(S, 0, Start).

%---------------------------------------------------------------------------%
% Transformations.

to_upper(S) = Upper :-
    to_upper(S, Upper).

:- pragma promise_equivalent_clauses(pred(to_upper/2)).
to_upper(S::in, Upper::uo) :-
    Upper = changed_case(S, 1).
to_upper(S::in, Upper::in) :-
    same(changed_case(S, 1), Upper).

to_lower(S) = Lower :-
    to_lower(S, Lower).

:- pragma promise_equivalent_clauses(pred(to_lower/2)).
to_lower(S::in, Lower::uo) :-
    Lower = changed_case(S, 0).
to_lower(S::in, Lower::in) :-
    same(changed_case(S, 0), Lower).

:- func changed_case(string, int) = string.
:- mode changed_case(in, in) = uo is det.
:- pragma foreign_proc("C",
    changed_case(S::in, Upper::in) = (R::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    R = merk_string_case(S, Upper);
").

pad_left(S, C, Width) = Padded :-
    pad_left(S, C, Width, Padded).

pad_left(S, C, Width, Padded) :-
    Count = Width - count_codepoints(S),
    ( if Count =< 0 then
        Padded = S
    else
        Padded = duplicate_char(C, Count) ++ S
    ).

pad_right(S, C, Width) = Padded :-
    pad_right(S, C, Width, Padded).

pad_right(S, C, Width, Padded) :-
    Count = Width - count_codepoints(S),
    ( if Count =< 0 then
        Padded = S
    else
        Padded = S ++ duplicate_char(C, Count)
    ).

strip(S) = stripped(S, 1, 1).
lstrip(S) = stripped(S, 1, 0).
rstrip(S) = stripped(S, 0, 1).

:- func stripped(string, int, int) = string.
:- pragma foreign_proc("C",
    stripped(S::in, Leading::in, Trailing::in) = (R::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    R = merk_string_strip(S, Leading, Trailing);
").

replace_all(S, Pattern, Replacement) = R :-
    replace_all(S, Pattern, Replacement, R).

:- pragma foreign_proc("C",
    replace_all(S::in, Pattern::in, Replacement::in, R::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    R = merk_string_replace_all(S, Pattern, Replacement);
").

%---------------------------------------------------------------------------%
% Folds over the code points.

foldl(F, S, A0) = A :-
    fold_from(F, S, 0, bytes(S), A0, A).

:- pred fold_from((func(char, A) = A)::in, string::in, int::in, int::in,
    A::in, A::out) is det.

fold_from(F, S, Offset, End, A0, A) :-
    ( if Offset >= End then
        A = A0
    else
        decode(S, Offset, C, Next),
        fold_from(F, S, Next, End, F(C, A0), A)
    ).

:- pragma promise_equivalent_clauses(pred(foldl/4)).
foldl(P::in(pred(in, di, uo) is det), S::in, A0::di, A::uo) :-
    fold_di(P, S, 0, bytes(S), A0, A).
foldl(P::in(pred(in, in, out) is det), S::in, A0::in, A::out) :-
    fold_det(P, S, 0, bytes(S), A0, A).
foldl(P::in(pred(in, in, out) is semidet), S::in, A0::in, A::out) :-
    fold_semidet(P, S, 0, bytes(S), A0, A).
foldl(P::in(pred(in, in, out) is nondet), S::in, A0::in, A::out) :-
    fold_nondet(P, S, 0, bytes(S), A0, A).
foldl(P::in(pred(in, in, out) is multi), S::in, A0::in, A::out) :-
    fold_multi(P, S, 0, bytes(S), A0, A).

:- pred fold_di(pred(char, A, A)::in(pred(in, di, uo) is det), string::in,
    int::in, int::in, A::di, A::uo) is det.

fold_di(P, S, Offset, End, !A) :-
    ( if Offset >= End then
        true
    else
        decode(S, Offset, C, Next),
        P(C, !A),
        fold_di(P, S, Next, End, !A)
    ).

:- pred fold_det(pred(char, A, A)::in(pred(in, in, out) is det), string::in,
    int::in, int::in, A::in, A::out) is det.

fold_det(P, S, Offset, End, !A) :-
    ( if Offset >= End then
        true
    else
        decode(S, Offset, C, Next),
        P(C, !A),
        fold_det(P, S, Next, End, !A)
    ).

:- pred fold_semidet(pred(char, A, A)::in(pred(in, in, out) is semidet),
    string::in, int::in, int::in, A::in, A::out) is semidet.

fold_semidet(P, S, Offset, End, !A) :-
    ( if Offset >= End then
        true
    else
        decode(S, Offset, C, Next),
        P(C, !A),
        fold_semidet(P, S, Next, End, !A)
    ).

:- pred fold_nondet(pred(char, A, A)::in(pred(in, in, out) is nondet),
    string::in, int::in, int::in, A::in, A::out) is nondet.

fold_nondet(P, S, Offset, End, !A) :-
    ( if Offset >= End then
        true
    else
        decode(S, Offset, C, Next),
        P(C, !A),
        fold_nondet(P, S, Next, End, !A)
    ).

:- pred fold_multi(pred(char, A, A)::in(pred(in, in, out) is multi),
    string::in, int::in, int::in, A::in, A::out) is multi.

fold_multi(P, S, Offset, End, !A) :-
    ( if Offset >= End then
        true
    else
        decode(S, Offset, C, Next),
        P(C, !A),
        fold_multi(P, S, Next, End, !A)
    ).

%---------------------------------------------------------------------------%
% Conversions to and from numbers.

:- pragma foreign_proc("C",
    to_int(S::in, I::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    I = 0;
    SUCCESS_INDICATOR = merk_string_to_int(S, &I);
").

det_to_int(S) = I :-
    ( if to_int(S, I0) then
        I = I0
    else
        unexpected("function `string.det_base_string_to_int'/2",
            "conversion failed")
    ).

:- pragma foreign_proc("C",
    to_float(S::in, F::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    double value = 0.0;
    SUCCESS_INDICATOR = merk_string_to_float(S, &value);
    F = value;
").

det_to_float(S) = F :-
    ( if to_float(S, F0) then
        F = F0
    else
        unexpected("function `string.det_to_float'/1", "conversion failed.")
    ).

char_to_string(C) = S :-
    char_to_string(C, S).

:- pragma promise_equivalent_clauses(pred(char_to_string/2)).
char_to_string(C::in, S::uo) :-
    from_char_list([C], S).
char_to_string(C::out, S::in) :-
    bytes(S) > 0,
    decode(S, 0, C, Next),
    Next = bytes(S).

from_char(C) = char_to_string(C).

int_to_string(I) = S :-
    int_to_string(I, S).

:- pragma foreign_proc("C",
    int_to_string(I::in, S::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    S = merk_string_from_int(I);
").

from_int(I) = int_to_string(I).

float_to_string(F) = S :-
    float_to_string(F, S).

:- pragma foreign_proc("C",
    float_to_string(F::in, S::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    S = merk_string_from_float(F);
").

from_float(F) = float_to_string(F).
