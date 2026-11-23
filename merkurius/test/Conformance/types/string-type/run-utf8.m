% MANUAL: String type
% A string is a sequence of characters in UTF-8 or UTF-16, the
% implementation's choice; it is UTF-8 here, as in mmc's C grades: length
% counts code units, here bytes, and count_codepoints the characters.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{7, 3}
% OUTPUT-NEXT:['a', 'é', '😀']
% OUTPUT-NEXT:"aé😀"
% OUTPUT-NOT:{{.}}

:- module run_utf8.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, list, string.

main(!IO) :-
    S = "a\u00e9\U0001F600",
    io.write_line({string.length(S) : int,
        string.count_codepoints(S) : int}, !IO),
    io.write_line(string.to_char_list(S) : list(char), !IO),
    io.write_line(S, !IO).
