% The sized integers' from_int, which fails for an int outside the type's
% range, and det_from_int, which throws mmc's software error for one: the
% int fits if converting the narrower value back gives it again, extended by
% its sign for a signed type and by zeros for an unsigned one; uint64 and
% uint take the ints that are not negative. For each type, its least and
% greatest values and the ints just outside them. The output is mmc's.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: not %t.O2 throw > %t.out 2> %t.err
% RUN: FileCheck %s --check-prefix=ERROR --match-full-lines --strict-whitespace < %t.err

% MERK-LABEL: merk.proc private @from_int.to_uint8_f1_0(%arg0: i64) -> i8
% MERK-NEXT: %[[LOW:.*]] = arith.trunci %arg0 : i64 to i8
% MERK-NEXT: %[[BACK:.*]] = arith.extui %[[LOW]] : i8 to i64
% MERK-NEXT: %[[OUT:.*]] = arith.cmpi ne, %[[BACK]], %arg0 : i64
% MERK-NEXT: merk.throw_if %[[OUT]], "Software Error: function `uint8.det_from_int'/1: cannot convert int to uint8"
% MERK-NEXT: merk.succeed %[[LOW]] : i8
% MERK-LABEL: merk.proc private @from_int.to_int16_2_0(%arg0: i64) -> i16
% MERK-NEXT: %[[LOW:.*]] = arith.trunci %arg0 : i64 to i16
% MERK-NEXT: %[[BACK:.*]] = arith.extsi %[[LOW]] : i16 to i64
% MERK-NEXT: %[[FITS:.*]] = arith.cmpi eq, %[[BACK]], %arg0 : i64
% MERK-NEXT: merk.guard %[[FITS]]
% MERK-NEXT: merk.succeed %[[LOW]] : i16

% OUTPUT:int8: [no, -128i8, 127i8, no]
% OUTPUT-NEXT:int16: [no, -32768i16, 32767i16, no]
% OUTPUT-NEXT:int32: [no, -2147483648i32, 2147483647i32, no]
% OUTPUT-NEXT:uint8: [no, 0u8, 255u8, no]
% OUTPUT-NEXT:uint16: [no, 0u16, 65535u16, no]
% OUTPUT-NEXT:uint32: [no, 0u32, 4294967295u32, no]
% OUTPUT-NEXT:uint64: [no, 0u64, 9223372036854775807u64]
% OUTPUT-NEXT:uint: [no, 0u, 9223372036854775807u]
% OUTPUT-NEXT:{-5i8, 300i16, 200u8, 5000000000u64}
% OUTPUT-NOT:{{.}}

% ERROR:Uncaught Merkurius exception:
% ERROR-NEXT:Software Error: function `uint8.det_from_int'/1: cannot convert int to uint8
% ERROR-NOT:{{.}}

:- module from_int.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- use_module int8, int16, int32, uint, uint8, uint16, uint32, uint64.

:- pred to_int16(int::in, int16::out) is semidet.
:- pragma no_inline(to_int16/2).
to_int16(I, X) :- int16.from_int(I, X).

:- func to_uint8(int) = uint8.
:- pragma no_inline(to_uint8/1).
to_uint8(I) = uint8.det_from_int(I).

:- pred line(string::in, pred(int, T)::in(pred(in, out) is semidet),
    list(int)::in, io::di, io::uo) is det.
line(Name, P, Is, !IO) :-
    io.write_string(Name ++ ": [", !IO),
    io.write_list(Is, ", ", show(P), !IO),
    io.write_string("]\n", !IO).

:- pred show(pred(int, T)::in(pred(in, out) is semidet), int::in,
    io::di, io::uo) is det.
show(P, I, !IO) :-
    ( if P(I, X) then io.write(X, !IO) else io.write_string("no", !IO) ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = ["throw"] then
        io.write_line(to_uint8(256), !IO)
    else
        line("int8", int8.from_int, [-129, -128, 127, 128], !IO),
        line("int16", to_int16, [-32769, -32768, 32767, 32768], !IO),
        line("int32", int32.from_int,
            [-2147483649, -2147483648, 2147483647, 2147483648], !IO),
        line("uint8", uint8.from_int, [-1, 0, 255, 256], !IO),
        line("uint16", uint16.from_int, [-1, 0, 65535, 65536], !IO),
        line("uint32", uint32.from_int, [-1, 0, 4294967295, 4294967296],
            !IO),
        line("uint64", uint64.from_int, [-1, 0, max_int], !IO),
        line("uint", uint.from_int, [-1, 0, max_int], !IO),
        io.write_line({int8.det_from_int(-5), int16.det_from_int(300),
            to_uint8(200), uint64.det_from_int(5000000000)}, !IO)
    ).
