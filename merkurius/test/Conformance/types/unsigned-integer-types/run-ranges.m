% MANUAL: Unsigned integer types
% uint8, uint16, uint32 and uint64 hold 0 to 2^N - 1; the extremes are
% literals, and arithmetic wraps modulo 2^N.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{0u8, 255u8, 0u8, 255u8}
% OUTPUT-NEXT:{0u16, 65535u16, 0u16, 65535u16}
% OUTPUT-NEXT:{0u32, 4294967295u32, 0u32, 4294967295u32}
% OUTPUT-NEXT:{0u64, 18446744073709551615u64, 0u64, 18446744073709551615u64}
% OUTPUT-NOT:{{.}}

:- module run_ranges.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module uint8, uint16, uint32, uint64.

main(!IO) :-
    io.write_line({0u8, 255u8, top8 + 1u8, zero8 - 1u8}, !IO),
    io.write_line({0u16, 65535u16, top16 + 1u16, zero16 - 1u16}, !IO),
    io.write_line({0u32, 4294967295u32, top32 + 1u32, zero32 - 1u32}, !IO),
    io.write_line({0u64, 18446744073709551615u64, top64 + 1u64,
        zero64 - 1u64}, !IO).

:- func top8 = uint8.
:- pragma no_inline(top8/0).
top8 = 255u8.
:- func zero8 = uint8.
:- pragma no_inline(zero8/0).
zero8 = 0u8.
:- func top16 = uint16.
:- pragma no_inline(top16/0).
top16 = 65535u16.
:- func zero16 = uint16.
:- pragma no_inline(zero16/0).
zero16 = 0u16.
:- func top32 = uint32.
:- pragma no_inline(top32/0).
top32 = 4294967295u32.
:- func zero32 = uint32.
:- pragma no_inline(zero32/0).
zero32 = 0u32.
:- func top64 = uint64.
:- pragma no_inline(top64/0).
top64 = 18446744073709551615u64.
:- func zero64 = uint64.
:- pragma no_inline(zero64/0).
zero64 = 0u64.
