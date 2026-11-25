% MANUAL: Signed integer types
% int is two's complement, at least 32 bits wide, its width the
% implementation's choice (64 here): min_int is -max_int - 1, and max_int +
% 1 wraps to min_int.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{-9223372036854775808, 9223372036854775807, 64}
% OUTPUT-NEXT:"equal"
% OUTPUT-NEXT:"wraps"
% OUTPUT-NOT:{{.}}

:- module run_int_width.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(!IO) :-
    io.write_line({int.min_int : int, int.max_int : int,
        int.bits_per_int : int}, !IO),
    io.write_line(( if -top - 1 = int.min_int then "equal" else "different" ),
        !IO),
    io.write_line(( if top + 1 = int.min_int then "wraps" else "no wrap" ),
        !IO).

:- func top = int.
:- pragma no_inline(top/0).
top = int.max_int : int.
