% io.write of operator atoms in each context, tuples of one element, binary prefix
% operators, and io.print and the _line forms.
% The expected output is mmc's for the same program.
%
% RUN: merkc %s -o %t && %t | diff %S/Inputs/io-write-operators.expected -
% RUN: env MERK_HEAP_CHUNK_BYTES=64 %t | diff %S/Inputs/io-write-operators.expected -
% RUN: merkc -O2 %s -o %t2 && %t2 | diff %S/Inputs/io-write-operators.expected -

:- module io_write_operators.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, pair.
:- type b ---> x ; y ; ('+') ; ('-') ; (not) ; (*) ; (!) ; (mod) ; (all) ; ('.') ; ('|') ; (or) ; (->) ; [] ; {} ; f(b) ; g(b, b) ; [b | b] ; (b + b) ; (b - b) ; (b = b) ; (b or b) ; (b -> b) ; (not b) ; - b ; \+ b ; (! b) ; all(b, b).
:- pred w(b::in, io::di, io::uo) is det.
w(X, !IO) :- io.write(X, !IO), io.nl(!IO).
:- pred t(T::in, io::di, io::uo) is det.
t(X, !IO) :- io.write(X, !IO), io.nl(!IO).
main(!IO) :-
    w((+), !IO), w(!, !IO), w((mod), !IO), w((all), !IO), w(('.'), !IO), w(('|'), !IO), w((or), !IO), w((->), !IO),
    w(g(!, (mod)), !IO), w(g((all), ('.')), !IO), w(g(('|'), (or)), !IO), w(f((->)), !IO),
    w([x | (+)], !IO), w([x | (x -> y)], !IO), w([(x -> y) | x], !IO), w([x | [y | (x -> y)]], !IO), w([x | (not)], !IO), w([(+)], !IO), w([(+), (-)], !IO),
    w(x or (+), !IO), w((+) or x, !IO), w(x -> (+), !IO), w((+) -> x, !IO), w(x = (+), !IO), w(not ((+)), !IO), w(\+ ((+)), !IO),
    w(! x, !IO), w(all(x, y), !IO), w(all(x, (+)), !IO), w(all((+), y), !IO),
    w(f(! x), !IO), w(f(all(x, y)), !IO), w(g(x -> y, x or y), !IO),
    more(x, y, (+), !IO),
    t(1 - 2, !IO), t(-1 - -2, !IO), t("a" - 'b', !IO),
    t('\x85\', !IO), t('\x9f\', !IO), t('\xa1\', !IO), t("\x85\", !IO), t("\x7e\", !IO), t('~', !IO),
    t('\\', !IO), t("\\", !IO), t('\'', !IO),
    t(1.0e100, !IO), t(1.0e15, !IO), t(1.0e16, !IO), t(123456789012345678.0, !IO), t(0.000001, !IO), t(1.5e-7, !IO), t(100.0, !IO), t(3.14159265358979, !IO), t(1.0e308, !IO), t(5.0e-324, !IO),
    io.print('c', !IO), io.nl(!IO), io.print("s\n", !IO), io.print({"s"}, !IO), io.nl(!IO), io.print(1.5, !IO), io.nl(!IO),
    io.write_line('c', !IO), io.write_line("s", !IO), io.print_line('c', !IO), io.print_line("s", !IO), io.print_line(1, !IO).

:- pred more(b::in, b::in, b::in, io::di, io::uo) is det.
more(X, Y, P, !IO) :-
    t({X -> Y}, !IO), t({X or Y}, !IO), t({P}, !IO), t({{X}}, !IO), t({{X, Y}}, !IO),
    t(f(Y) - {X}, !IO), t({X} - f(Y), !IO), t({(X -> Y), Y}, !IO), t({P, Y}, !IO).
