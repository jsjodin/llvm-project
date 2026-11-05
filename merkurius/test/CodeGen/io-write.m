% io.write, io.print and their _line forms on each kind of value, at T through a
% type_info's descriptor (design document, "Type descriptors"): integers with
% their suffixes, floats, characters and strings with their escapes, constructors,
% lists, tuples, arrays, and operators with their priorities.
% The expected output is mmc's for the same program.
%
% RUN: merkc %s -o %t && %t | diff %S/Inputs/io-write.expected -
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | diff %S/Inputs/io-write.expected -
% RUN: merkc -O2 %s -o %t2 && %t2 | diff %S/Inputs/io-write.expected -

:- module io_write.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, list, float, int8, uint8, uint16, int64, uint, maybe, pair.

:- type colour ---> red ; green ; 'Blue' ; 'with space' ; [] ; '[|]' ; {} ; 'a.b' ; (+) ; (=>).
:- type t ---> k(int, string) ; aa ; m(t) ; e(float, colour) ; 'Odd Name'(int) ; (int + int) ; - int ; nest(list(int)) ; u(float).
:- type wrap ---> wrap(int).
:- type wf ---> wf(float).
:- type unit ---> unit.
:- type tr(T) ---> leaf ; node(tr(T), T, tr(T)).
:- type rec ---> rec(a :: int, b :: string).
:- type narrow ---> narrow(int8, uint8, uint16, colour).
:- type eqv == list(int).
:- type ops ---> (int * int) ; (t + t) ; (ops or ops) ; (ops -> ops) ; (ops and ops) ; (ops - ops) ; (int mod int) ; \ ops ; \+ ops ; [ops | ops] ; (ops = ops) ; {ops, ops}.

:- pred w(T::in, io::di, io::uo) is det.
w(X, !IO) :- io.write(X, !IO), io.nl(!IO).

main(!IO) :-
    w(1, !IO), w(-1, !IO), w(1u, !IO), w(-3i8, !IO), w(200u8, !IO), w(60000u16, !IO), w(-5i64, !IO),
    w(0xffffffffffffffffu, !IO),
    w(1.0, !IO), w(-1.5, !IO), w(1.0e30, !IO), w(0.1, !IO), w(1.0e-5, !IO), w(float.infinity, !IO), w(-0.0, !IO),
    w(unchecked_quotient(0.0, 0.0), !IO),
    w('a', !IO), w('\n', !IO), w('\'', !IO), w('"', !IO), w('é', !IO), w('😀', !IO), w(' ', !IO), w('\t', !IO), wch('\\', !IO), w('\x01\', !IO), w('(', !IO), w('Z', !IO), w('1', !IO), w('\x7f\', !IO),
    w("abc", !IO), w("a\"b", !IO), w("a\nb", !IO), w("a\\b", !IO), w("é😀", !IO), w("\t", !IO), w("\x01\", !IO), w("'", !IO), w("", !IO), w("\x7f\", !IO),
    terms(!IO),
    w(wrap(1), !IO), w(wf(2.0), !IO), w(unit, !IO),
    w(node(leaf, 1, leaf), !IO), w(node(node(leaf, "x", leaf), "y", leaf), !IO),
    w(rec(1, "z"), !IO),
    w(narrow(-1i8, 2u8, 3u16, green), !IO),
    w([] : list(int), !IO), w([1], !IO), w([1, 2, 3], !IO), w(["a", "b"], !IO), w([[1], [2, 3]], !IO), w([red, green], !IO), w([1.0], !IO), w(['a'], !IO),
    w({1, "a"}, !IO), w({1}, !IO), w({1, {2, 3}}, !IO), w({red, [1], "s", 'c', 1.5}, !IO),
    w(yes(1), !IO), w(no : maybe(int), !IO), w(1 - 2, !IO), w("a" - [1], !IO), w((1 - 2) - 3, !IO), w(1 - (2 - 3), !IO),
    array.init(2, 7, A), w(A, !IO), array.init(0, 7, A0), w(A0, !IO), array.init(2, "s", S), w(S, !IO), w([A], !IO), array.init(1, red, C), w(C, !IO),
    ops(!IO),
    % other entry points
    io.write_line(1, !IO), io.write_line("s", !IO), io.print(1, !IO), io.nl(!IO), io.print("s", !IO), io.nl(!IO), io.print('c', !IO), io.nl(!IO), io.print([1], !IO), io.nl(!IO), io.print(["s"], !IO), io.nl(!IO),
    io.print_line("s", !IO), io.print_line(k(1, "a"), !IO).
:- func colour_nil = colour.
colour_nil = [].

:- pred o(ops::in, io::di, io::uo) is det.
o(X, !IO) :- w(X, !IO).
:- func oa = ops.
oa = 1 * 2.
:- func ob = ops.
ob = 3 * 4.
:- func oc = ops.
oc = 5 * 6.
:- pred ops(io::di, io::uo) is det.
ops(!IO) :-
    o(oa, !IO), o(aa + aa, !IO), o(m(aa) + (1 + 2), !IO), o(oa or ob, !IO), o(oa -> ob, !IO), o(oa and ob, !IO),
    o((oa - ob) - oc, !IO), o(oa - (ob - oc), !IO), o(3 mod 4, !IO), o(\ oa, !IO), o(\+ oa, !IO),
    o([oa | ob], !IO), o(oa = ob, !IO), o({oa, ob}, !IO),
    o((oa -> ob) or oc, !IO), o((oa or ob) -> oc, !IO), o((oa - ob) = (oc -> oa), !IO),
    o(\ (oa - ob), !IO), o(\ (\ oa), !IO), o(\+ (\ oa), !IO), o((\ oa) - ob, !IO), o((\+ oa) - ob, !IO),
    o(oa - (\+ ob), !IO), o((oa and ob) and oc, !IO), o(oa and (ob and oc), !IO), o((oa = ob) = oc, !IO), o(oa = (ob = oc), !IO).

:- pred wc(colour::in, io::di, io::uo) is det.
wc(X, !IO) :- w(X, !IO).
:- pred wt(t::in, io::di, io::uo) is det.
wt(X, !IO) :- w(X, !IO).
:- pred terms(io::di, io::uo) is det.
terms(!IO) :-
    wc(red, !IO), wc('Blue', !IO), wc('with space', !IO), wc(colour_nil, !IO), wc('[|]', !IO), wc({}, !IO), wc('a.b', !IO), wc((+), !IO), wc((=>), !IO),
    wt(k(1, "oa"), !IO), wt(aa, !IO), wt(m(aa), !IO), wt(e(1.5, red), !IO), wt('Odd Name'(1), !IO),
    wt(1 + 2, !IO), N3 = 3, wt(- N3, !IO), wt(nest([1, 2]), !IO), wt(u(-1.0), !IO).

:- pred wch(character::in, io::di, io::uo) is det.
wch(C, !IO) :- w(C, !IO).
