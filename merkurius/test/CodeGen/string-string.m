% string.string writes a value as io.write does, into a string: each line
% is a value written by both. The output is mmc's for the same program.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:42 | 42
% OUTPUT-NEXT:-3 | -3
% OUTPUT-NEXT:1.5 | 1.5
% OUTPUT-NEXT:3.0 | 3.0
% OUTPUT-NEXT:1e+20 | 1e+20
% OUTPUT-NEXT:infinity | infinity
% OUTPUT-NEXT:"ab\"c\n\té" | "ab\"c\n\té"
% OUTPUT-NEXT:'a' | 'a'
% OUTPUT-NEXT:'\'' | '\''
% OUTPUT-NEXT:'\n' | '\n'
% OUTPUT-NEXT:[1, 2, 3] | [1, 2, 3]
% OUTPUT-NEXT:[] | []
% OUTPUT-NEXT:{1, "a", 'b'} | {1, "a", 'b'}
% OUTPUT-NEXT:yes(2) | yes(2)
% OUTPUT-NEXT:1 - "x" | 1 - "x"
% OUTPUT-NEXT:1 + 2 | 1 + 2
% OUTPUT-NEXT:- 5 | - 5
% OUTPUT-NEXT:f(1 + 2, - 3) | f(1 + 2, - 3)
% OUTPUT-NEXT:'a b' | 'a b'
% OUTPUT-NEXT:[] | []
% OUTPUT-NEXT:x("y", 'z') | x("y", 'z')
% OUTPUT-NEXT:array([1, 2]) | array([1, 2])
% OUTPUT-NEXT:-1i8 | -1i8
% OUTPUT-NEXT:7u | 7u
% OUTPUT-NEXT:["é", "日本"] | ["é", "日本"]
% OUTPUT-NEXT:'<<function>>' | '<<function>>'
% OUTPUT-NEXT:{{\[\[}}1], [2, 3]] | {{\[\[}}1], [2, 3]]
% OUTPUT-NEXT:"h�" | "h�"

:- module string_string.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, char, list, string, maybe, pair, array, float, int8, uint.
:- type op ---> (int + int) ; - int ; f(op, op) ; 'a b' ; [] ; x(string, char).
:- pred w(T::in, io::di, io::uo) is det.
w(X, !IO) :-
    io.write(X, !IO), io.write_string(" | ", !IO),
    io.write_string(string.string(X), !IO), io.nl(!IO).
main(!IO) :-
    w(42, !IO), w(-3, !IO), w(1.5, !IO), w(3.0, !IO), w(1.0e20, !IO),
    w(float.infinity, !IO), w("ab\"c\n\té", !IO), w('a', !IO), w('\'', !IO),
    w('\n', !IO), w([1, 2, 3], !IO), w([] : list(int), !IO), w({1, "a", 'b'}, !IO),
    w(yes(2) : maybe(int), !IO), w(1 - "x", !IO), w((1 + 2) : op, !IO), w('-'(5) : op, !IO),
    w(f((1 + 2) : op, '-'(3) : op), !IO), w('a b', !IO), w([] : op, !IO), w(x("y", 'z'), !IO),
    w(array([1, 2]), !IO), w(-1i8, !IO), w(7u, !IO), w(["é", "日本"], !IO),
    w((func(X) = int.(X + 1)) : (func(int) = int), !IO), w([[1], [2, 3]], !IO),
    w(string.left("héllo", 2) : string, !IO).
