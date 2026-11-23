% MANUAL: Unsigned integer types
% The width of uint is the width of int: max_uint is 2 * max_int + 1, and
% the bit counts agree. The width itself, 64 bits here, is the
% implementation's choice, at least 32.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:18446744073709551615u
% OUTPUT-NEXT:9223372036854775807
% OUTPUT-NEXT:"equal"
% OUTPUT-NEXT:{64, 64}
% OUTPUT-NOT:{{.}}

:- module run_uint_width.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, uint.

main(!IO) :-
    io.write_line(uint.max_uint : uint, !IO),
    io.write_line(int.max_int : int, !IO),
    io.write_line(
        ( if uint.cast_from_int(int.max_int) * 2u + 1u = uint.max_uint
        then "equal" else "different" ) : string, !IO),
    io.write_line({int.bits_per_int : int, uint.bits_per_uint : int}, !IO).
