% Integer and float literals, and when '-' is part of a number. Each item is
% read by merkc and by Mercury's own term parser, which must agree.
%
% REQUIRES: mmc
% RUN: %compare-terms %s

X = - 1.
X = -1.
X = - 1.5.
X = -(1).
X = - (1).
X = -(-(1)).
X = a - 1.
X = a -1.
X = a-1.
X = [-1].
X = "\x41\\101\".
X = 0'a.
X = 0' .
X = 0''.
X = 0xFF.
X = 0x7fffffffffffffff.
X = 0x8000000000000000.
X = 0xffffffffffffffffu.
X = 127i8.
X = 128i8.
X = -128i8.
X = -129i8.
X = 32767i16.
X = 32768i16.
X = 65535u16.
X = 65536u16.
X = 2147483647i32.
X = 2147483648i32.
X = 4294967295u32.
X = 4294967296u32.
X = 9223372036854775807.
X = 9223372036854775808.
X = -9223372036854775808.
X = 18446744073709551615u.
X = 18446744073709551616u.
X = -1u.
X = -0u.
X = 1e10.
X = 1.0E-5.
X = 1.5e+3.
X = 0.1.
X = 1_000.000_1.
X = 1.0e309.
X = '-' 1.
X = - /* c */ 1.
X = -
    1.
X = - 1 + 2.
X = - 1 * 2.
X = - 1.5e3.
X = - 0'a.
X = - 0xff.
X = - 1i8.
X = - 128i8.
X = -(1, 2).
X = - (1, 2).
X = 1.0e-400.
X = -1.0e309.
X = - 1u8.
