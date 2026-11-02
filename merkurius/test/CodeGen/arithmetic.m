% The arithmetic of int, uint, the sized integers and float, which code
% generation builds in as arith operations, and the comparisons, which are
% tests. Integers wrap; a division checks for a zero divisor, and one of
% the least integer by -1 wraps, as mmc folds it; a shift checks its
% amount. The output is mmc's for the same program, but for the second and
% third lines, of min_int by -1, where mmc's program dies of SIGFPE (see
% the design document's register).
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK-LABEL: merk.proc private @arithmetic.quotient_f2_0(%arg0: i64, %arg1: i64) -> i64
% MERK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
% MERK-NEXT: %[[IS_ZERO:.*]] = arith.cmpi eq, %arg1, %[[ZERO]] : i64
% MERK-NEXT: merk.throw_if %[[IS_ZERO]], "domain_error(\22int.\\'//\\': division by zero\22)"
% MERK-NEXT: %[[MINUS_ONE:.*]] = arith.constant -1 : i64
% MERK-NEXT: %[[BY_MINUS_ONE:.*]] = arith.cmpi eq, %arg1, %[[MINUS_ONE]] : i64
% MERK-NEXT: %[[ONE:.*]] = arith.constant 1 : i64
% MERK-NEXT: %[[DIVISOR:.*]] = arith.select %[[BY_MINUS_ONE]], %[[ONE]], %arg1 : i64
% MERK-NEXT: %[[Q:.*]] = arith.divsi %arg0, %[[DIVISOR]] : i64
% MERK: %[[NEGATED:.*]] = arith.subi %{{.*}}, %arg0 : i64
% MERK-NEXT: %[[RESULT:.*]] = arith.select %[[BY_MINUS_ONE]], %[[NEGATED]], %[[Q]] : i64
% MERK: merk.succeed %[[RESULT]] : i64

% LLVM-LABEL: define internal tailcc i64 @arithmetic.quotient_f2_0(i64 %0, i64 %1) {
% LLVM-NEXT: %[[IS_ZERO:.*]] = icmp eq i64 %1, 0
% LLVM-NEXT: br i1 %[[IS_ZERO]], label %[[THROW:.*]], label %[[DIVIDE:.*]]
% LLVM: [[THROW]]:
% LLVM-NEXT: call void @merk_uncaught_exception(ptr @merk.str.{{[0-9]+}})
% LLVM-NEXT: unreachable
% LLVM: [[DIVIDE]]:
% LLVM: sdiv i64 %0,
% LLVM: declare void @merk_uncaught_exception(ptr) #[[ATTRS:[0-9]+]]
% LLVM: attributes #[[ATTRS]] = { noreturn nounwind }

% OUTPUT:-3
% OUTPUT-NEXT:-9223372036854775808
% OUTPUT-NEXT:0
% OUTPUT-NEXT:9 5 14 3 1 3 1 2 7 5 -8 -7 2 7 >>=\= 28 1
% OUTPUT-NEXT:-5 -9 -14 -3 -1 -4 1 0 -5 -5 6 7 -7 2 <=<\= -28 -2
% OUTPUT-NEXT:5 9 -14 -3 1 -4 -1 6 -1 -7 -8 -7 -2 7 >>=\=
% OUTPUT-NEXT:-9 -5 14 3 -1 3 -1 -8 -1 7 6 7 -7 -2 <=<\=
% OUTPUT-NEXT:9 3 18 2 0 2 0 2 7 5 -7 -6 3 6 >>=\= 48 0
% OUTPUT-NEXT:-3 -9 -18 -2 0 -2 0 2 -5 -7 5 6 -6 3 <=<\= -48 -1
% OUTPUT-NEXT:5 -5 0 0 0 0 0 0 5 5 -1 0 0 5 <=<\= 0 0
% OUTPUT-NEXT:5 5 0 0 5 5 -6 -5 0 5 >>=\= 5 5
% OUTPUT-NEXT:-9223372036854775808 9223372036854775806 9223372036854775807 9223372036854775807 0 9223372036854775807 0 1 9223372036854775807 9223372036854775806 -9223372036854775808 -9223372036854775807 1 9223372036854775807 >>=\= -2 4611686018427387903
% OUTPUT-NEXT:-9223372036854775745 9223372036854775745 -9223372036854775808 -146402730743726600 -8 -146402730743726601 55 0 -9223372036854775745 -9223372036854775745 9223372036854775807 -9223372036854775808 -9223372036854775808 63 <=<\= 0 -1
% OUTPUT-NEXT:62 -64 -63 0 -1 -1 62 63 -1 -64 0 1 -1 63 <=<\= -9223372036854775808 -1
% OUTPUT-NEXT:6 0 9 1 0 1 0 3 3 0 -4 -3 3 3 =<>== 24 0
% OUTPUT-NEXT:9.0 5.0 14.0 3.5 -7.0 7 >>=
% OUTPUT-NEXT:-5.5 -9.5 -15.0 -3.75 7.5 -7 <=<
% OUTPUT-NEXT:1.0 1.0 0.0 -1.0 1 >>=
% OUTPUT-NEXT:5.0 0.0 6.25 1.0 -2.5 2 =<>=
% OUTPUT-NEXT:2e+300 0.0 infinity 1.0 -1e+300 -9223372036854775808 =<>=
% OUTPUT-NEXT:7.0
% OUTPUT-NEXT:3 18446744073709551615 0 2305843009213693951 <
% OUTPUT-NEXT:-128 -3 4 5
% OUTPUT-NOT:{{.}}

:- module arithmetic.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, uint, int8, uint8, float.
:- pred show(int::in, int::in, io::di, io::uo) is det.
show(A, B, !IO) :-
    io.write_int(A + B, !IO), io.write_char(' ', !IO),
    io.write_int(A - B, !IO), io.write_char(' ', !IO),
    io.write_int(A * B, !IO), io.write_char(' ', !IO),
    ( if B = 0 then
        true
    else
        io.write_int(A // B, !IO), io.write_char(' ', !IO),
        io.write_int(A rem B, !IO), io.write_char(' ', !IO),
        io.write_int(A div B, !IO), io.write_char(' ', !IO),
        io.write_int(A mod B, !IO), io.write_char(' ', !IO)
    ),
    io.write_int(A /\ B, !IO), io.write_char(' ', !IO),
    io.write_int(A \/ B, !IO), io.write_char(' ', !IO),
    io.write_int(xor(A, B), !IO), io.write_char(' ', !IO),
    io.write_int(\ A, !IO), io.write_char(' ', !IO),
    io.write_int(- A, !IO), io.write_char(' ', !IO),
    io.write_int(min(A, B), !IO), io.write_char(' ', !IO),
    io.write_int(max(A, B), !IO), io.write_char(' ', !IO),
    ( if A < B then io.write_string("<", !IO) else true ),
    ( if A =< B then io.write_string("=<", !IO) else true ),
    ( if A > B then io.write_string(">", !IO) else true ),
    ( if A >= B then io.write_string(">=", !IO) else true ),
    ( if A = B then io.write_string("=", !IO) else true ),
    ( if A \= B then io.write_string("\\=", !IO) else true ),
    ( if B >= 0, B < 64 then
        io.write_char(' ', !IO),
        io.write_int(A << B, !IO), io.write_char(' ', !IO),
        io.write_int(A >> B, !IO)
    else
        true
    ),
    io.nl(!IO).
:- pred showf(float::in, float::in, io::di, io::uo) is det.
showf(A, B, !IO) :-
    io.write_float(A + B, !IO), io.write_char(' ', !IO),
    io.write_float(A - B, !IO), io.write_char(' ', !IO),
    io.write_float(A * B, !IO), io.write_char(' ', !IO),
    ( if B = 0.0 then true else io.write_float(A / B, !IO), io.write_char(' ', !IO) ),
    io.write_float(- A, !IO), io.write_char(' ', !IO),
    io.write_int(truncate_to_int(A), !IO), io.write_char(' ', !IO),
    ( if A < B then io.write_string("<", !IO) else true ),
    ( if A =< B then io.write_string("=<", !IO) else true ),
    ( if A > B then io.write_string(">", !IO) else true ),
    ( if A >= B then io.write_string(">=", !IO) else true ),
    io.nl(!IO).
:- func quotient(int, int) = int.
quotient(A, B) = A // B.
main(!IO) :-
    io.write_int(quotient(-7, 2), !IO), io.nl(!IO),
    io.write_int(quotient(min_int, -1), !IO), io.nl(!IO),
    io.write_int(min_int rem (-1 + max_int - max_int), !IO), io.nl(!IO),
    show(7, 2, !IO), show(-7, 2, !IO), show(7, -2, !IO), show(-7, -2, !IO),
    show(6, 3, !IO), show(-6, 3, !IO), show(0, 5, !IO), show(5, 0, !IO),
    show(max_int, 1, !IO), show(min_int, 63, !IO),
    show(-1, 63, !IO), show(3, 3, !IO),
    showf(7.0, 2.0, !IO), showf(-7.5, 2.0, !IO), showf(1.0, 0.0, !IO),
    showf(2.5, 2.5, !IO), showf(1.0e300, 1.0e300, !IO),
    io.write_float(float(7), !IO), io.nl(!IO),
    io.write_uint(7u // 2u, !IO), io.write_char(' ', !IO),
    io.write_uint(max_uint, !IO), io.write_char(' ', !IO),
    io.write_uint(max_uint + 1u, !IO), io.write_char(' ', !IO),
    io.write_uint(max_uint >> 3, !IO), io.write_char(' ', !IO),
    ( if 3u < max_uint then io.write_string("<", !IO) else true ),
    io.nl(!IO),
    io.write_int8(int8.max_int8 + 1i8, !IO), io.write_char(' ', !IO),
    io.write_int8(-7i8 // 2i8, !IO), io.write_char(' ', !IO),
    io.write_uint8(250u8 + 10u8, !IO), io.write_char(' ', !IO),
    io.write_int(abs(-5), !IO),
    io.nl(!IO).
