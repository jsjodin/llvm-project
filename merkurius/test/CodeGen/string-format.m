% string.format: each conversion with its flags, widths and precisions,
% from the format and from inputs, of each type of input, on ASCII and
% multi-byte UTF-8, and infinities and NaNs. The output is mmc's for the
% same program, but for a negative precision from an input, which is C's:
% mmc writes the C format it made (design document, "Where mmc and the
% manual disagree").
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:plain => [plain]
% OUTPUT-NEXT:%% => [%]
% OUTPUT-NEXT:%d|%i => [42|-7]
% OUTPUT-NEXT:%5d|%-5d|%05d|%+d|% d|%+05d => [   42|42   |00042|+42| 42|-0042]
% OUTPUT-NEXT:%.3d|%8.3d|%-8.3d|%.0d => [007|     007|-007    |]
% OUTPUT-NEXT:%x|%X|%o|%u => [ff|FF|10|42]
% OUTPUT-NEXT:%#x|%#X|%#o|%#o => [0xff|0XFF|010|0]
% OUTPUT-NEXT:%x|%u|%o => [ffffffffffffffff|18446744073709551615|1777777777777777777777]
% OUTPUT-NEXT:%x|%u|%X => [ff|42|FFFFFFFFFFFFFFFF]
% OUTPUT-NEXT:%d|%d|%d|%d => [-128|-300|70000|-9223372036854775808]
% OUTPUT-NEXT:%u|%x|%u|%u => [255|ffff|4294967295|18446744073709551615]
% OUTPUT-NEXT:%x|%o => [ff|177777]
% OUTPUT-NEXT:%p|%p => [0xff|0x10]
% OUTPUT-NEXT:%c|%3c|%-3c| => [a|  é|日  |]
% OUTPUT-NEXT:%s|%5s|%-5s|%.2s|%5.1s| => [ab|   ab|ab   |ab|    x|]
% OUTPUT-NEXT:%5s|%-5s|%.2s|%.1s| => [    é|日本   |hé|日|]
% OUTPUT-NEXT:%f|%.2f|%10.3f|%-10.1f|%+f|% f|%010.2f => [3.141590|3.14|     3.142|2.5       |+1.000000| 1.000000|-000002.50]
% OUTPUT-NEXT:%e|%.2e|%E|%#.0e|%.0e => [1.234568e+04|1.23e+04|1.230000E-04|5.e+00|5e+00]
% OUTPUT-NEXT:%g|%G|%g|%#g|%.3g|%g => [0.0001|1E-10|1.23457e+08|1.00000|3.14|100000]
% OUTPUT-NEXT:%f|%f|%e|%g => [infinity|-infinity|nan|infinity]
% OUTPUT-NEXT:%#.0f|%.0f|%.0f|%.0f => [2.|2|4|-0]
% OUTPUT-NEXT:%*d|%-*d|%.*f|%*.*f => [    42|42    |3.14|   2.718]
% OUTPUT-NEXT:%*d => [42    ]
% OUTPUT-NEXT:%d%%%s => [5%x]
% OUTPUT-NEXT:%5%| => [%|]
% OUTPUT-NEXT:%-+ 0#5d| => [+3   |]
% OUTPUT-NEXT:%8f|%-10e|%+g|%G|%E|%05f|% f => [infinity|-infinity |infinity|INFINITY|NAN|infinity|infinity]
% OUTPUT-NEXT:%5.3s|%-4c|%04c|%+s|%05s => [  日本語|é   |   x|a|   ab]
% OUTPUT-NEXT:%.0s|%.10s|%3.0d|%-03d|%.3x|%#.3o|%#5x|%-#8X => [|ab|   |5  |00a|010|    0|0XFF    ]
% OUTPUT-NEXT:%x|%o|%u|%u => [ffffffffffffffff|37777777777|255|200]
% OUTPUT-NEXT:%12f|%-12f|%012f|%+12f|%012e|%-+12g|%12g => [    infinity|infinity    |   -infinity|   -infinity|         nan|infinity    |         nan]
% OUTPUT-NEXT:%c|%c => [|😀]
% OUTPUT-NEXT:%f|%.1e|%g|%g|%g|%g => [0.000000|0.0e+00|0|1e+15|1e+16|1e-05]
% OUTPUT-NEXT:%.17g|%.0g|%#.3g|%#x|%#o => [0.10000000000000001|1e+02|1.00|0xffffffffffffffff|01777777777777777777777]
% OUTPUT-NEXT:%i|%.2i|%+i|%+u|% u|%+x => [3|03|+3|3|3|3]
% OUTPUT-NEXT:%*s|%-*s|%.*s|%*c => [   é|é   |日|  x]
% OUTPUT-NEXT:%.*d|%.*f => [5|1.500000]
% OUTPUT-NEXT:a%sb%sc => [abéc]
% OUTPUT-NEXT:%f => [1000000000000000052504760255204420248704468581108159154915854115511802457988908195786371375080447864043704443832883878176942523235360430575644792184786706982848387200926575803737830233794788090059368953234970799945081119038967640880074652742780142494579258788820056842838115669472196386865459400540160.000000]
% OUTPUT-NEXT:%.20f => [0.10000000000000000555]
% OUTPUT-NEXT:%10.4e|%-12.3E| => [-1.2345e+03|5.000E-01   |]
% OUTPUT-NEXT:%c => [
% OUTPUT-NEXT:]
% OUTPUT-NEXT:%s => [tab	here]

:- module string_format.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, int8, int16, int32, int64, list, string,
    uint, uint8, uint16, uint32, uint64.

:- pred f(string::in, list(poly_type)::in, io::di, io::uo) is det.
f(Format, Args, !IO) :-
    io.write_string(Format, !IO),
    io.write_string(" => [", !IO),
    io.write_string(string.format(Format, Args), !IO),
    io.write_string("]\n", !IO).

main(!IO) :-
    f("plain", [], !IO),
    f("%%", [], !IO),
    f("%d|%i", [i(42), i(-7)], !IO),
    f("%5d|%-5d|%05d|%+d|% d|%+05d", [i(42), i(42), i(42), i(42), i(42), i(-42)], !IO),
    f("%.3d|%8.3d|%-8.3d|%.0d", [i(7), i(7), i(-7), i(0)], !IO),
    f("%x|%X|%o|%u", [i(255), i(255), i(8), i(42)], !IO),
    f("%#x|%#X|%#o|%#o", [i(255), i(255), i(8), i(0)], !IO),
    f("%x|%u|%o", [i(-1), i(-1), i(-1)], !IO),
    f("%x|%u|%X", [u(255u), u(42u), u(18446744073709551615u)], !IO),
    f("%d|%d|%d|%d", [i8(-128i8), i16(-300i16), i32(70000i32), i64(-9223372036854775808i64)], !IO),
    f("%u|%x|%u|%u", [u8(255u8), u16(65535u16), u32(4294967295u32), u64(18446744073709551615u64)], !IO),
    f("%x|%o", [i8(-1i8), i16(-1i16)], !IO),
    f("%p|%p", [i(255), u(16u)], !IO),
    f("%c|%3c|%-3c|", [c('a'), c('é'), c('日')], !IO),
    f("%s|%5s|%-5s|%.2s|%5.1s|", [s("ab"), s("ab"), s("ab"), s("abcdef"), s("xyz")], !IO),
    f("%5s|%-5s|%.2s|%.1s|", [s("é"), s("日本"), s("héllo"), s("日本")], !IO),
    f("%f|%.2f|%10.3f|%-10.1f|%+f|% f|%010.2f", [f(3.14159), f(3.14159), f(3.14159), f(2.5), f(1.0), f(1.0), f(-2.5)], !IO),
    f("%e|%.2e|%E|%#.0e|%.0e", [f(12345.678), f(12345.678), f(0.000123), f(5.0), f(5.0)], !IO),
    f("%g|%G|%g|%#g|%.3g|%g", [f(0.0001), f(1.0e-10), f(123456789.0), f(1.0), f(3.14159), f(100000.0)], !IO),
    f("%f|%f|%e|%g", [f(float.infinity), f(-float.infinity), f(float.infinity * 0.0), f(float.infinity)], !IO),
    f("%#.0f|%.0f|%.0f|%.0f", [f(2.0), f(2.5), f(3.5), f(-0.5)], !IO),
    f("%*d|%-*d|%.*f|%*.*f", [i(6), i(42), i(6), i(42), i(2), f(3.14159), i(8), i(3), f(2.71828)], !IO),
    f("%*d", [i(-6), i(42)], !IO),
    f("%d%%%s", [i(5), s("x")], !IO),
    f("%5%|", [], !IO),
    f("%-+ 0#5d|", [i(3)], !IO),
    f("%8f|%-10e|%+g|%G|%E|%05f|% f", [f(float.infinity), f(-float.infinity), f(float.infinity), f(float.infinity), f(float.infinity * 0.0), f(float.infinity), f(float.infinity)], !IO),
    f("%5.3s|%-4c|%04c|%+s|%05s", [s("日本語x"), c('é'), c('x'), s("a"), s("ab")], !IO),
    f("%.0s|%.10s|%3.0d|%-03d|%.3x|%#.3o|%#5x|%-#8X", [s("abc"), s("ab"), i(0), i(5), i(10), i(8), i(0), i(255)], !IO),
    f("%x|%o|%u|%u", [i64(-1i64), i32(-1i32), i8(-1i8), u8(200u8)], !IO),
    f("%12f|%-12f|%012f|%+12f|%012e|%-+12g|%12g", [f(float.infinity), f(float.infinity), f(-float.infinity), f(-float.infinity), f(float.infinity * 0.0), f(float.infinity), f(-float.infinity * 0.0)], !IO),
    f("%c|%c", [c('\x80\'), c('😀')], !IO),
    f("%f|%.1e|%g|%g|%g|%g", [f(-0.0), f(-0.0), f(-0.0), f(1.0e15), f(1.0e16), f(0.00001)], !IO),
    f("%.17g|%.0g|%#.3g|%#x|%#o", [f(0.1), f(123.0), f(1.0), i(-1), i(-1)], !IO),
    f("%i|%.2i|%+i|%+u|% u|%+x", [i(3), i(3), i(3), i(3), i(3), i(3)], !IO),
    f("%*s|%-*s|%.*s|%*c", [i(4), s("é"), i(4), s("é"), i(1), s("日本"), i(3), c('x')], !IO),
    f("%.*d|%.*f", [i(-1), i(5), i(-1), f(1.5)], !IO),
    f("a%sb%sc", [s(""), s("é")], !IO),
    f("%f", [f(1.0e300)], !IO),
    f("%.20f", [f(0.1)], !IO),
    f("%10.4e|%-12.3E|", [f(-1234.5), f(0.5)], !IO),
    f("%c", [c('\n')], !IO),
    f("%s", [s("tab\there")], !IO).
