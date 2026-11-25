% MANUAL: Tokens
% String literals: doubled quotes, the backslash escapes, hexadecimal,
% octal and Unicode escapes, an escaped newline and an embedded one.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:["say \"hi\"", "\a\b\r\f\t\n\v", "\\ \' \"", "Ab", "Ab", "é😀", "one two", "line\nbreak"]
% OUTPUT-NEXT:[8, 7, 5, 2, 2, 2, 7, 10]
% OUTPUT-NOT:{{.}}

:- module run_strings.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- func cps(string) = int.
cps(S) = string.count_codepoints(S).

main(!IO) :-
    L = ["say ""hi""", "\a\b\r\f\t\n\v", "\\ \' \"",
        "\x41\\x62\", "\101\\142\", "\u00e9\U0001F600",
        "one \
two", "line
break"],
    io.write(L, !IO), io.nl(!IO),
    io.write(list.map(cps, L), !IO), io.nl(!IO).
