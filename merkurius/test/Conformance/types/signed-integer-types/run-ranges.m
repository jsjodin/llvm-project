% MANUAL: Signed integer types
% int8, int16, int32 and int64 hold exactly the ranges their widths give,
% in two's complement: the extremes are literals, and arithmetic on them
% wraps, as the manual leaves overflow to two's-complement representation.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{-128i8, 127i8}
% OUTPUT-NEXT:{-32768i16, 32767i16}
% OUTPUT-NEXT:{-2147483648i32, 2147483647i32}
% OUTPUT-NEXT:{-9223372036854775808i64, 9223372036854775807i64}
% OUTPUT-NEXT:{-128i8, 127i8}
% OUTPUT-NEXT:{-32768i16, 32767i16}
% OUTPUT-NEXT:{-2147483648i32, 2147483647i32}
% OUTPUT-NEXT:{-9223372036854775808i64, 9223372036854775807i64}
% OUTPUT-NOT:{{.}}

:- module run_ranges.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, int8, int16, int32, int64.

main(!IO) :-
    io.write_line({-128i8, 127i8}, !IO),
    io.write_line({-32768i16, 32767i16}, !IO),
    io.write_line({-2147483648i32, 2147483647i32}, !IO),
    io.write_line({-9223372036854775808i64, 9223372036854775807i64}, !IO),
    io.write_line({top8 + 1i8, bottom8 - 1i8}, !IO),
    io.write_line({top16 + 1i16, bottom16 - 1i16}, !IO),
    io.write_line({top32 + 1i32, bottom32 - 1i32}, !IO),
    io.write_line({top64 + 1i64, bottom64 - 1i64}, !IO).

:- func top8 = int8.
:- pragma no_inline(top8/0).
top8 = 127i8.
:- func bottom8 = int8.
:- pragma no_inline(bottom8/0).
bottom8 = -128i8.
:- func top16 = int16.
:- pragma no_inline(top16/0).
top16 = 32767i16.
:- func bottom16 = int16.
:- pragma no_inline(bottom16/0).
bottom16 = -32768i16.
:- func top32 = int32.
:- pragma no_inline(top32/0).
top32 = 2147483647i32.
:- func bottom32 = int32.
:- pragma no_inline(bottom32/0).
bottom32 = -2147483648i32.
:- func top64 = int64.
:- pragma no_inline(top64/0).
top64 = 9223372036854775807i64.
:- func bottom64 = int64.
:- pragma no_inline(bottom64/0).
bottom64 = -9223372036854775808i64.
