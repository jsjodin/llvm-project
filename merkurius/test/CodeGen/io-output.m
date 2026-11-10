% The output predicates of io that the compiled standard library has so
% far. The output is mmc's for the same program: a float with the fewest
% significant digits, from 15 to 17, that read back as it, with .0 if that
% has no fraction or exponent; a character in UTF-8.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --match-full-lines --strict-whitespace

% CHECK:-42
% CHECK-NEXT:9223372036854775807
% CHECK-NEXT:é😀x
% CHECK-NEXT:1.0
% CHECK-NEXT:0.1
% CHECK-NEXT:1e+100
% CHECK-NEXT:3.14159
% CHECK-NEXT:100.0
% CHECK-NEXT:1.5e-10
% CHECK-NEXT:1.2345678901234568e+17
% CHECK-NEXT:1.7976931348623157e+308
% CHECK-NEXT:2.2250738585072014e-308
% CHECK-NEXT:2e+15
% CHECK-NEXT:1e+16
% CHECK-NEXT:tab	here
% CHECK-EMPTY:

:- module io_output.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

main(!IO) :-
    io.write_int(-42, !IO), io.nl(!IO),
    io.write_int(9223372036854775807, !IO), io.nl(!IO),
    io.write_char('é', !IO), io.write_char('\x1F600\', !IO),
    io.write_char('x', !IO), io.nl(!IO),
    io.write_float(1.0, !IO), io.nl(!IO),
    io.write_float(0.1, !IO), io.nl(!IO),
    io.write_float(1.0e100, !IO), io.nl(!IO),
    io.write_float(3.14159, !IO), io.nl(!IO),
    io.write_float(100.0, !IO), io.nl(!IO),
    io.write_float(1.5e-10, !IO), io.nl(!IO),
    io.write_float(123456789012345680.0, !IO), io.nl(!IO),
    io.write_float(1.7976931348623157e308, !IO), io.nl(!IO),
    io.write_float(2.2250738585072014e-308, !IO), io.nl(!IO),
    io.write_float(2.0e15, !IO), io.nl(!IO),
    io.write_float(1.0e16, !IO), io.nl(!IO),
    io.write_string("tab\there\n", !IO).
