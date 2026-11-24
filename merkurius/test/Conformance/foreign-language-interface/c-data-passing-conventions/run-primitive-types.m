% MANUAL: C data passing conventions
% Each primitive type is passed as its C type, at the sizes the manual
% gives, and a string comes back from C.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:-5 -8 -16 -32 -64 5 8 16 32 64 1.50 233 str
% OUTPUT-NOT:{{.}}

:- module run_primitive_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64.
:- import_module char, string.
:- pragma foreign_decl("C", local, "#include <string.h>\n#include <stdio.h>").
:- pred prims(int::in, int8::in, int16::in, int32::in, int64::in, uint::in,
    uint8::in, uint16::in, uint32::in, uint64::in, float::in, char::in,
    string::in, string::out) is det.
:- pragma foreign_proc("C", prims(I::in, I8::in, I16::in, I32::in, I64::in,
    U::in, U8::in, U16::in, U32::in, U64::in, F::in, C::in, S::in, Out::out),
    [promise_pure, will_not_call_mercury],
"
    MR_Integer i = I; int8_t i8 = I8; int16_t i16 = I16; int32_t i32 = I32;
    int64_t i64 = I64; MR_Unsigned u = U; uint8_t u8 = U8; uint16_t u16 = U16;
    uint32_t u32 = U32; uint64_t u64 = U64; MR_Float f = F; MR_Char c = C;
    MR_String s = S;
    char buf[256];
    snprintf(buf, sizeof buf, ""%ld %d %d %d %lld %lu %u %u %u %llu %.2f %d %s"",
        (long) i, i8, i16, i32, (long long) i64, (unsigned long) u, u8, u16,
        u32, (unsigned long long) u64, f, (int) c, s);
    Out = strdup(buf);
").

main(!IO) :-
    prims(-5, -8i8, -16i16, -32i32, -64i64, 5u, 8u8, 16u16, 32u32, 64u64,
        1.5, 'é', "str", S),
    io.write_string(S ++ "\n", !IO).
