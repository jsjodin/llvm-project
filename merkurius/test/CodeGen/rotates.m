% Rotations of unsigned fixed-width integers: rotate_left, rotate_right and
% their unchecked forms, for each width. The amounts are not constants, as K
% is 0 at run time. A checked rotation by the width or more throws, with
% mmc's message; which/1 picks the case. Nothing catches it, so the program
% writes the exception to the standard error and exits with status 1.
% The output and messages are mmc's.
%
% RUN: merkc %s -o %t.0 && %t.0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.0.O2 && %t.0.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: sed 's/^which(0)\./which(1)./' %s > %t.1.m && merkc %t.1.m -o %t.1
% RUN: not %t.1 2> %t.1.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-1 --match-full-lines --strict-whitespace < %t.1.err
% RUN: sed 's/^which(0)\./which(2)./' %s > %t.2.m && merkc -O2 %t.2.m -o %t.2
% RUN: not %t.2 2> %t.2.err | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=ERROR-2 --match-full-lines --strict-whitespace < %t.2.err

:- module rotates.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string, uint, uint8, uint16, uint32, uint64.

:- pred which(int::out) is det.
which(0).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    % K is 0 at run time, so the amounts are not constants.
    K = uint.cast_from_int(list.length(Args)),
    A32 = 0x87654321_u32,
    io.format("%x %x %x %x\n",
        [i(uint32.cast_to_int(rotate_left(A32, K + 4u))),
         i(uint32.cast_to_int(rotate_right(A32, K + 4u))),
         i(uint32.cast_to_int(unchecked_rotate_left(A32, K))),
         i(uint32.cast_to_int(unchecked_rotate_right(A32, K + 31u)))], !IO),
    io.format("%x %x %x %x\n",
        [i(uint8.cast_to_int(rotate_left(0x81u8, K + 1u))),
         i(uint8.cast_to_int(rotate_right(0x81u8, K + 7u))),
         i(uint16.cast_to_int(unchecked_rotate_left(0x8001u16, K + 15u))),
         i(uint16.cast_to_int(rotate_right(0x1234u16, K + 8u)))], !IO),
    io.format("%x %x\n",
        [i(uint64.cast_to_int(rotate_right(0x1u64, K + 1u) >> 32)),
         i(uint64.cast_to_int(unchecked_rotate_left(0xF000000000000001u64,
            K + 4u)))], !IO),
    io.format("%x %x\n",
        [i(uint32.cast_to_int(rotate_left(A32, 4u))),
         i(uint8.cast_to_int(unchecked_rotate_right(0x81u8, 1u)))], !IO),
    ( if which(1) then
        io.format("%x\n", [i(uint32.cast_to_int(rotate_left(A32, K + 32u)))],
            !IO)
    else if which(2) then
        io.format("%x\n", [i(uint8.cast_to_int(rotate_right(0x81u8, K + 8u)))],
            !IO)
    else
        true
    ).

% OUTPUT:76543218 18765432 87654321 eca8643
% OUTPUT-NEXT:3 3 c000 3412
% OUTPUT-NEXT:80000000 1f
% OUTPUT-NEXT:76543218 c0
% OUTPUT-NOT:{{.}}

% ERROR-1:Uncaught Merkurius exception:
% ERROR-1-NEXT:Software Error: function `uint32.rotate_left'/2: rotate amount exceeds 31 bits

% ERROR-2:Uncaught Merkurius exception:
% ERROR-2-NEXT:Software Error: function `uint8.rotate_right'/2: rotate amount exceeds 7 bits
