% The string module, compiled from library/string.m on runtime/string.cpp:
% every declaration but format and string/1, in each of its modes, on
% ASCII, multi-byte UTF-8 and empty strings and at the edges of their
% indices. Indices count bytes, and a byte not part of a well-formed
% sequence reads as U+FFFD, as io.write writes it, as in mmc. The output is
% mmc's for the same program.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT: = {0, 0}
% OUTPUT-NEXT:  is_empty = yes
% OUTPUT-NEXT:  to_char_list = []
% OUTPUT-NEXT:  to_rev_char_list = []
% OUTPUT-NEXT:a = {1, 1}
% OUTPUT-NEXT:  is_empty = no
% OUTPUT-NEXT:  to_char_list = ['a']
% OUTPUT-NEXT:  to_rev_char_list = ['a']
% OUTPUT-NEXT:abc = {3, 3}
% OUTPUT-NEXT:  is_empty = no
% OUTPUT-NEXT:  to_char_list = ['a', 'b', 'c']
% OUTPUT-NEXT:  to_rev_char_list = ['c', 'b', 'a']
% OUTPUT-NEXT:héllo = {6, 5}
% OUTPUT-NEXT:  is_empty = no
% OUTPUT-NEXT:  to_char_list = ['h', 'é', 'l', 'l', 'o']
% OUTPUT-NEXT:  to_rev_char_list = ['o', 'l', 'l', 'é', 'h']
% OUTPUT-NEXT:日本語 = {9, 3}
% OUTPUT-NEXT:  is_empty = no
% OUTPUT-NEXT:  to_char_list = ['日', '本', '語']
% OUTPUT-NEXT:  to_rev_char_list = ['語', '本', '日']
% OUTPUT-NEXT:a b  c  = {7, 7}
% OUTPUT-NEXT:  is_empty = no
% OUTPUT-NEXT:  to_char_list = ['a', ' ', 'b', ' ', ' ', 'c', ' ']
% OUTPUT-NEXT:  to_rev_char_list = [' ', 'c', ' ', ' ', 'b', ' ', 'a']
% OUTPUT-NEXT: 	 x
% OUTPUT-NEXT:y 
% OUTPUT-NEXT: = {8, 8}
% OUTPUT-NEXT:  is_empty = no
% OUTPUT-NEXT:  to_char_list = [' ', '\t', ' ', 'x', '\n', 'y', ' ', '\n']
% OUTPUT-NEXT:  to_rev_char_list = ['\n', ' ', 'y', '\n', 'x', ' ', '\t', ' ']
% OUTPUT-NEXT:😀z = {5, 2}
% OUTPUT-NEXT:  is_empty = no
% OUTPUT-NEXT:  to_char_list = ['😀', 'z']
% OUTPUT-NEXT:  to_rev_char_list = ['z', '😀']
% OUTPUT-NEXT:++ = "abcd"
% OUTPUT-NEXT:append = "abé"
% OUTPUT-NEXT:append_list = "abcé"
% OUTPUT-NEXT:append_list empty = ""
% OUTPUT-NEXT:join_list = "a, b, c"
% OUTPUT-NEXT:join_list one = "a"
% OUTPUT-NEXT:join_list empty = ""
% OUTPUT-NEXT:append(in, in, in) = yes
% OUTPUT-NEXT:append(in, in, in) no = no
% OUTPUT-NEXT:append(in, uo, in) = yes("cd")
% OUTPUT-NEXT:append(in, uo, in) no = no
% OUTPUT-NEXT:append(uo, in, in) = yes("ab")
% OUTPUT-NEXT:append(uo, in, in) no = no
% OUTPUT-NEXT:append(out, out, in) = [{"", "aé"}, {"a", "é"}, {"aé", ""}]
% OUTPUT-NEXT:append(out, out, in) empty = [{"", ""}]
% OUTPUT-NEXT:duplicate_char = "ééé"
% OUTPUT-NEXT:duplicate_char 0 = ""
% OUTPUT-NEXT:duplicate_char -1 = ""
% OUTPUT-NEXT:from_char_list = "aé日"
% OUTPUT-NEXT:from_char_list(out, in) = ['h', 'é']
% OUTPUT-NEXT:to_char_list(uo, in) = "xÿ"
% OUTPUT-NEXT:to_rev_char_list(uo, in) = "ÿx"
% OUTPUT-NEXT:first_char(in, in, in) = yes
% OUTPUT-NEXT:first_char(in, uo, in) = yes('é')
% OUTPUT-NEXT:first_char(in, in, uo) = yes("bc")
% OUTPUT-NEXT:first_char(in, uo, uo) = yes({'é', "bc"})
% OUTPUT-NEXT:first_char(in, uo, uo) empty = no
% OUTPUT-NEXT:first_char(uo, in, in) = "éxy"
% OUTPUT-NEXT:char_to_string = "日"
% OUTPUT-NEXT:char_to_string(out, in) = yes('é')
% OUTPUT-NEXT:char_to_string(out, in) no = no
% OUTPUT-NEXT:char_to_string(out, in) empty = no
% OUTPUT-NEXT:from_char = "x"
% OUTPUT-NEXT:index = yes('é')
% OUTPUT-NEXT:index inside a code point = yes('�')
% OUTPUT-NEXT:index past = no
% OUTPUT-NEXT:index negative = no
% OUTPUT-NEXT:det_index = 'c'
% OUTPUT-NEXT:contains_char = yes
% OUTPUT-NEXT:contains_char no = no
% OUTPUT-NEXT:foldl = 102
% OUTPUT-NEXT:foldl pred = ['é', 'b', 'a']
% OUTPUT-NEXT:words = ["the", "quick", "brown", "fox"]
% OUTPUT-NEXT:words empty = []
% OUTPUT-NEXT:words blank = []
% OUTPUT-NEXT:split_at_char = ["a", "", "b", ""]
% OUTPUT-NEXT:split_at_char empty = [""]
% OUTPUT-NEXT:split_at_char none = ["ab"]
% OUTPUT-NEXT:split_into_lines = ["a", "b", "", "c"]
% OUTPUT-NEXT:split_into_lines trailing = ["a", "b"]
% OUTPUT-NEXT:split_into_lines empty = []
% OUTPUT-NEXT:split_into_lines newline = [""]
% OUTPUT-NEXT:prefix(in, in) = yes
% OUTPUT-NEXT:prefix(in, in) no = no
% OUTPUT-NEXT:prefix(in, out) = ["", "a", "aé", "aéb"]
% OUTPUT-NEXT:suffix(in, in) = yes
% OUTPUT-NEXT:suffix(in, out) = ["", "aéb", "b", "éb"]
% OUTPUT-NEXT:remove_prefix = yes("c")
% OUTPUT-NEXT:remove_prefix no = no
% OUTPUT-NEXT:remove_suffix = yes("a")
% OUTPUT-NEXT:remove_suffix no = no
% OUTPUT-NEXT:to_upper = "HéLLO WORLD 1"
% OUTPUT-NEXT:to_lower = "hÉllo world 1"
% OUTPUT-NEXT:to_upper(in, in) = yes
% OUTPUT-NEXT:to_lower(in, in) no = no
% OUTPUT-NEXT:pad_left = "...ab"
% OUTPUT-NEXT:pad_left short = "abcdef"
% OUTPUT-NEXT:pad_left utf8 = "~~é"
% OUTPUT-NEXT:pad_right = "ab..."
% OUTPUT-NEXT:pad_right utf8 = "日本**"
% OUTPUT-NEXT:strip = "a b"
% OUTPUT-NEXT:lstrip = "a b  "
% OUTPUT-NEXT:rstrip = "  a b"
% OUTPUT-NEXT:strip all = ""
% OUTPUT-NEXT:replace_all = "a--b--c"
% OUTPUT-NEXT:replace_all empty pattern = "-a-b-c-"
% OUTPUT-NEXT:replace_all none = "abc"
% OUTPUT-NEXT:replace_all overlapping = "ba"
% OUTPUT-NEXT:to_int = yes(-42)
% OUTPUT-NEXT:to_int plus = yes(7)
% OUTPUT-NEXT:to_int bad = no
% OUTPUT-NEXT:to_int empty = no
% OUTPUT-NEXT:to_int minus = no
% OUTPUT-NEXT:to_int space = no
% OUTPUT-NEXT:to_int max = yes(9223372036854775807)
% OUTPUT-NEXT:to_int over = no
% OUTPUT-NEXT:to_int min = yes(-9223372036854775808)
% OUTPUT-NEXT:to_int under = no
% OUTPUT-NEXT:to_int underscore = no
% OUTPUT-NEXT:to_int hex = no
% OUTPUT-NEXT:to_int leading zeros = yes(7)
% OUTPUT-NEXT:det_to_int = 123
% OUTPUT-NEXT:to_float = yes(3.25)
% OUTPUT-NEXT:to_float exponent = yes(1000.0)
% OUTPUT-NEXT:to_float int = yes(12.0)
% OUTPUT-NEXT:to_float bad = no
% OUTPUT-NEXT:to_float dot = yes(0.5)
% OUTPUT-NEXT:to_float trailing dot = yes(5.0)
% OUTPUT-NEXT:to_float space = no
% OUTPUT-NEXT:to_float inf = yes(infinity)
% OUTPUT-NEXT:to_float nan = yes(nan)
% OUTPUT-NEXT:to_float hex = yes(8.0)
% OUTPUT-NEXT:to_float empty = no
% OUTPUT-NEXT:to_float underscore = no
% OUTPUT-NEXT:det_to_float = -0.5
% OUTPUT-NEXT:int_to_string = "-12345"
% OUTPUT-NEXT:int_to_string pred = "77"
% OUTPUT-NEXT:from_int = "-9223372036854775808"
% OUTPUT-NEXT:float_to_string = "0.1"
% OUTPUT-NEXT:float_to_string big = "1e+20"
% OUTPUT-NEXT:float_to_string negative = "-2.5"
% OUTPUT-NEXT:float_to_string whole = "3.0"
% OUTPUT-NEXT:float_to_string small = "1e-07"
% OUTPUT-NEXT:from_float = "0.3333333333333333"
% OUTPUT-NEXT:between = "é"
% OUTPUT-NEXT:between negative = "hel"
% OUTPUT-NEXT:between past = "lo"
% OUTPUT-NEXT:between reversed = ""
% OUTPUT-NEXT:left = "h�"
% OUTPUT-NEXT:left negative = ""
% OUTPUT-NEXT:left past = "hello"
% OUTPUT-NEXT:right = "lo"
% OUTPUT-NEXT:right past = "hello"
% OUTPUT-NEXT:right negative = ""
% OUTPUT-NEXT:sub_string_search = yes(3)
% OUTPUT-NEXT:sub_string_search empty = yes(0)
% OUTPUT-NEXT:sub_string_search no = no

:- module strings.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, float, int, list, maybe, solutions, string.

    % A value.
:- pred v(string::in, T::in, io::di, io::uo) is det.
v(Label, X, !IO) :-
    io.write_string(Label, !IO), io.write_string(" = ", !IO),
    io.write(X, !IO), io.nl(!IO).

    % A string.
:- pred vs(string::in, string::in, io::di, io::uo) is det.
vs(Label, X, !IO) :-
    v(Label, X, !IO).

    % Whether a test succeeds.
:- pred t(string::in, (pred)::in((pred) is semidet), io::di, io::uo) is det.
t(Label, P, !IO) :-
    ( if P then R = "yes" else R = "no" ),
    io.write_string(Label, !IO), io.write_string(" = ", !IO),
    io.write_string(R, !IO), io.nl(!IO).

    % A semidet output.
:- pred m(string::in, pred(T)::in(pred(out) is semidet), io::di, io::uo)
    is det.
m(Label, P, !IO) :-
    ( if P(X) then R = yes(X) else R = no ),
    v(Label, R, !IO).

    % All the solutions.
:- pred a(string::in, pred(T)::in(pred(out) is multi), io::di, io::uo)
    is det.
a(Label, P, !IO) :-
    solutions(P, L),
    v(Label, L, !IO).

:- func samples = list(string).
samples = ["", "a", "abc", "héllo", "日本語", "a b  c ", " \t x\ny \n", "😀z"].

main(!IO) :-
    list.foldl(each, samples, !IO),
    appending(!IO),
    characters(!IO),
    splitting(!IO),
    affixes(!IO),
    transforming(!IO),
    numbers(!IO),
    substrings(!IO).

:- pred each(string::in, io::di, io::uo) is det.
each(S, !IO) :-
    v(S, {string.length(S) : int, string.count_codepoints(S) : int}, !IO),
    t("  is_empty", ((pred) is semidet :- string.is_empty(S)), !IO),
    v("  to_char_list", string.to_char_list(S) : list(char), !IO),
    v("  to_rev_char_list", string.to_rev_char_list(S) : list(char), !IO).

:- pred appending(io::di, io::uo) is det.
appending(!IO) :-
    vs("++", "ab" ++ "cd", !IO),
    vs("append", string.append("ab", "é"), !IO),
    vs("append_list", string.append_list(["a", "", "bc", "é"]), !IO),
    vs("append_list empty", string.append_list([]), !IO),
    vs("join_list", string.join_list(", ", ["a", "b", "c"]), !IO),
    vs("join_list one", string.join_list(", ", ["a"]), !IO),
    vs("join_list empty", string.join_list(", ", []), !IO),
    t("append(in, in, in)", ((pred) is semidet :- string.append("ab", "c", "abc")), !IO),
    t("append(in, in, in) no", ((pred) is semidet :- string.append("ab", "d", "abc")), !IO),
    m("append(in, uo, in)", (pred(S::out) is semidet :- string.append("ab", S, "abcd")), !IO),
    m("append(in, uo, in) no", (pred(S::out) is semidet :- string.append("x", S, "abcd")), !IO),
    m("append(uo, in, in)", (pred(S::out) is semidet :- string.append(S, "cd", "abcd")), !IO),
    m("append(uo, in, in) no", (pred(S::out) is semidet :- string.append(S, "x", "abcd")), !IO),
    a("append(out, out, in)", (pred(P::out) is multi :- string.append(A, B, "aé"), P = {A, B}), !IO),
    a("append(out, out, in) empty", (pred(P::out) is multi :- string.append(A, B, ""), P = {A, B}), !IO).

:- pred characters(io::di, io::uo) is det.
characters(!IO) :-
    vs("duplicate_char", string.duplicate_char('é', 3), !IO),
    vs("duplicate_char 0", string.duplicate_char('x', 0), !IO),
    vs("duplicate_char -1", string.duplicate_char('x', -1), !IO),
    vs("from_char_list", string.from_char_list(['a', 'é', '日']), !IO),
    string.from_char_list(Cs, "hé"),
    v("from_char_list(out, in)", Cs, !IO),
    string.to_char_list(Back, ['x', 'ÿ']),
    v("to_char_list(uo, in)", Back, !IO),
    string.to_rev_char_list(RBack, ['x', 'ÿ']),
    v("to_rev_char_list(uo, in)", RBack, !IO),
    t("first_char(in, in, in)", ((pred) is semidet :- string.first_char("abc", 'a', "bc")), !IO),
    m("first_char(in, uo, in)", (pred(C::out) is semidet :- string.first_char("ébc", C, "bc")), !IO),
    m("first_char(in, in, uo)", (pred(R::out) is semidet :- string.first_char("ébc", 'é', R)), !IO),
    m("first_char(in, uo, uo)", (pred(P::out) is semidet :- string.first_char("ébc", C, R), P = {C, R}), !IO),
    m("first_char(in, uo, uo) empty", (pred(P::out) is semidet :- string.first_char("", C, R), P = {C, R}), !IO),
    string.first_char(FC, 'é', "xy"),
    v("first_char(uo, in, in)", FC, !IO),
    vs("char_to_string", string.char_to_string('日'), !IO),
    m("char_to_string(out, in)", (pred(C::out) is semidet :- string.char_to_string(C, "é")), !IO),
    m("char_to_string(out, in) no", (pred(C::out) is semidet :- string.char_to_string(C, "ab")), !IO),
    m("char_to_string(out, in) empty", (pred(C::out) is semidet :- string.char_to_string(C, "")), !IO),
    vs("from_char", string.from_char('x'), !IO),
    m("index", (pred(C::out) is semidet :- string.index("héllo", 1, C)), !IO),
    m("index inside a code point", (pred(C::out) is semidet :- string.index("héllo", 2, C)), !IO),
    m("index past", (pred(C::out) is semidet :- string.index("abc", 3, C)), !IO),
    m("index negative", (pred(C::out) is semidet :- string.index("abc", -1, C)), !IO),
    v("det_index", string.det_index("abc", 2) : char, !IO),
    t("contains_char", ((pred) is semidet :- string.contains_char("héllo", 'é')), !IO),
    t("contains_char no", ((pred) is semidet :- string.contains_char("hello", 'x')), !IO),
    v("foldl", string.foldl(func(C, N) = (if C = 'é' then N + 100 else N + 1), "aéb", 0) : int, !IO),
    string.foldl((pred(C::in, L0::in, L::out) is det :- L = [C | L0]), "abé", [], FL),
    v("foldl pred", FL, !IO).

:- pred splitting(io::di, io::uo) is det.
splitting(!IO) :-
    v("words", string.words("  the quick\tbrown\n fox ") : list(string), !IO),
    v("words empty", string.words("") : list(string), !IO),
    v("words blank", string.words("  ") : list(string), !IO),
    v("split_at_char", string.split_at_char(',', "a,,b,") : list(string), !IO),
    v("split_at_char empty", string.split_at_char(',', "") : list(string), !IO),
    v("split_at_char none", string.split_at_char(',', "ab") : list(string), !IO),
    v("split_into_lines", string.split_into_lines("a\nb\n\nc") : list(string), !IO),
    v("split_into_lines trailing", string.split_into_lines("a\nb\n") : list(string), !IO),
    v("split_into_lines empty", string.split_into_lines("") : list(string), !IO),
    v("split_into_lines newline", string.split_into_lines("\n") : list(string), !IO).

:- pred affixes(io::di, io::uo) is det.
affixes(!IO) :-
    t("prefix(in, in)", ((pred) is semidet :- string.prefix("abc", "ab")), !IO),
    t("prefix(in, in) no", ((pred) is semidet :- string.prefix("abc", "b")), !IO),
    a("prefix(in, out)", (pred(P::out) is multi :- string.prefix("aéb", P)), !IO),
    t("suffix(in, in)", ((pred) is semidet :- string.suffix("abc", "bc")), !IO),
    a("suffix(in, out)", (pred(P::out) is multi :- string.suffix("aéb", P)), !IO),
    m("remove_prefix", (pred(S::out) is semidet :- string.remove_prefix("ab", "abc", S)), !IO),
    m("remove_prefix no", (pred(S::out) is semidet :- string.remove_prefix("x", "abc", S)), !IO),
    m("remove_suffix", (pred(S::out) is semidet :- string.remove_suffix("abc", "bc", S)), !IO),
    m("remove_suffix no", (pred(S::out) is semidet :- string.remove_suffix("abc", "x", S)), !IO).

:- pred transforming(io::di, io::uo) is det.
transforming(!IO) :-
    vs("to_upper", string.to_upper("héllo World 1"), !IO),
    vs("to_lower", string.to_lower("HÉLLO World 1"), !IO),
    t("to_upper(in, in)", ((pred) is semidet :- string.to_upper("ab", "AB")), !IO),
    t("to_lower(in, in) no", ((pred) is semidet :- string.to_lower("AB", "Ab")), !IO),
    vs("pad_left", string.pad_left("ab", '.', 5), !IO),
    vs("pad_left short", string.pad_left("abcdef", '.', 3), !IO),
    vs("pad_left utf8", string.pad_left("é", '~', 3), !IO),
    vs("pad_right", string.pad_right("ab", '.', 5), !IO),
    vs("pad_right utf8", string.pad_right("日本", '*', 4), !IO),
    vs("strip", string.strip("  \t a b \n "), !IO),
    vs("lstrip", string.lstrip("  a b  "), !IO),
    vs("rstrip", string.rstrip("  a b  "), !IO),
    vs("strip all", string.strip("   "), !IO),
    vs("replace_all", string.replace_all("a.b.c", ".", "--"), !IO),
    vs("replace_all empty pattern", string.replace_all("abc", "", "-"), !IO),
    vs("replace_all none", string.replace_all("abc", "x", "-"), !IO),
    vs("replace_all overlapping", string.replace_all("aaa", "aa", "b"), !IO).

:- pred numbers(io::di, io::uo) is det.
numbers(!IO) :-
    m("to_int", (pred(I::out) is semidet :- string.to_int("-42", I)), !IO),
    m("to_int plus", (pred(I::out) is semidet :- string.to_int("+7", I)), !IO),
    m("to_int bad", (pred(I::out) is semidet :- string.to_int("4x", I)), !IO),
    m("to_int empty", (pred(I::out) is semidet :- string.to_int("", I)), !IO),
    m("to_int minus", (pred(I::out) is semidet :- string.to_int("-", I)), !IO),
    m("to_int space", (pred(I::out) is semidet :- string.to_int(" 4", I)), !IO),
    m("to_int max", (pred(I::out) is semidet :- string.to_int("9223372036854775807", I)), !IO),
    m("to_int over", (pred(I::out) is semidet :- string.to_int("9223372036854775808", I)), !IO),
    m("to_int min", (pred(I::out) is semidet :- string.to_int("-9223372036854775808", I)), !IO),
    m("to_int under", (pred(I::out) is semidet :- string.to_int("-9223372036854775809", I)), !IO),
    m("to_int underscore", (pred(I::out) is semidet :- string.to_int("1_000", I)), !IO),
    m("to_int hex", (pred(I::out) is semidet :- string.to_int("0x1f", I)), !IO),
    m("to_int leading zeros", (pred(I::out) is semidet :- string.to_int("007", I)), !IO),
    v("det_to_int", string.det_to_int("123") : int, !IO),
    m("to_float", (pred(F::out) is semidet :- string.to_float("3.25", F)), !IO),
    m("to_float exponent", (pred(F::out) is semidet :- string.to_float("1e3", F)), !IO),
    m("to_float int", (pred(F::out) is semidet :- string.to_float("12", F)), !IO),
    m("to_float bad", (pred(F::out) is semidet :- string.to_float("1.5x", F)), !IO),
    m("to_float dot", (pred(F::out) is semidet :- string.to_float(".5", F)), !IO),
    m("to_float trailing dot", (pred(F::out) is semidet :- string.to_float("5.", F)), !IO),
    m("to_float space", (pred(F::out) is semidet :- string.to_float(" 1.5", F)), !IO),
    m("to_float inf", (pred(F::out) is semidet :- string.to_float("inf", F)), !IO),
    m("to_float nan", (pred(F::out) is semidet :- string.to_float("nan", F)), !IO),
    m("to_float hex", (pred(F::out) is semidet :- string.to_float("0x1p3", F)), !IO),
    m("to_float empty", (pred(F::out) is semidet :- string.to_float("", F)), !IO),
    m("to_float underscore", (pred(F::out) is semidet :- string.to_float("1_0.5", F)), !IO),
    v("det_to_float", string.det_to_float("-0.5") : float, !IO),
    vs("int_to_string", string.int_to_string(-12345), !IO),
    string.int_to_string(77, IS),
    v("int_to_string pred", IS, !IO),
    vs("from_int", string.from_int(min_int), !IO),
    vs("float_to_string", string.float_to_string(0.1), !IO),
    vs("float_to_string big", string.float_to_string(1.0e20), !IO),
    vs("float_to_string negative", string.float_to_string(-2.5), !IO),
    vs("float_to_string whole", string.float_to_string(3.0), !IO),
    vs("float_to_string small", string.float_to_string(1.0e-7), !IO),
    vs("from_float", string.from_float(1.0 / 3.0), !IO).

:- pred substrings(io::di, io::uo) is det.
substrings(!IO) :-
    vs("between", string.between("héllo", 1, 3), !IO),
    vs("between negative", string.between("hello", -2, 3), !IO),
    vs("between past", string.between("hello", 3, 10), !IO),
    vs("between reversed", string.between("hello", 3, 1), !IO),
    vs("left", string.left("héllo", 2), !IO),
    vs("left negative", string.left("hello", -1), !IO),
    vs("left past", string.left("hello", 9), !IO),
    vs("right", string.right("héllo", 2), !IO),
    vs("right past", string.right("hello", 9), !IO),
    vs("right negative", string.right("hello", -1), !IO),
    m("sub_string_search", (pred(I::out) is semidet :- string.sub_string_search("héllo", "llo", I)), !IO),
    m("sub_string_search empty", (pred(I::out) is semidet :- string.sub_string_search("abc", "", I)), !IO),
    m("sub_string_search no", (pred(I::out) is semidet :- string.sub_string_search("abc", "x", I)), !IO).
