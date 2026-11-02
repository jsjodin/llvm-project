% A procedure with a foreign_proc for C runs its C code. Code generation
% puts the code in a C function, which the procedure calls: its inputs are
% the function's parameters, and it returns the only output, or stores
% several through pointers after the inputs. A variable of type io.state
% is an MR_Word. The module records its C code, after its own foreign_decl
% for C; clang compiles it, and its functions are linked into the module.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s -o %t.mlir
% RUN: FileCheck %s --check-prefix=MERK < %t.mlir
% RUN: FileCheck %s --check-prefix=C < %t.mlir
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines

% C: merk.foreign_code = "{{.*}}typedef intptr_t MR_Integer;
% C-SAME: #include <string.h>
% C-SAME: \0AMR_Integer merk_foreign_foreign__proc_Dsucc__2__0(MR_Integer X) {\0A  MR_Integer Y;\0A  {\0A#line {{[0-9]+}} \22{{.*}}foreign-proc.m\22\0AY = X + 1;\0A  }\0A  return Y;\0A}\0A
% C-SAME: \0Avoid merk_foreign_foreign__proc_Ddivide__4__0(MR_Integer A, MR_Integer B, MR_Integer *merk_out_Q, MR_Integer *merk_out_R) {
% C-SAME: *merk_out_Q = Q;\0A  *merk_out_R = R;\0A}
% C-SAME: \0AMR_Integer merk_foreign_foreign__proc_Dlength__f1__0(MR_String S) {
% C-SAME: \0Avoid merk_foreign_foreign__proc_Dshow__3__0(MR_Integer N) {\0A  MR_Word _IO0 = 0;\0A  (void)_IO0;\0A  MR_Word _IO = 0;\0A  (void)_IO;\0A

% MERK-LABEL: merk.proc private @foreign_proc.succ_2_0(%arg0: i64) -> i64
% MERK-NEXT: %[[Y:.*]] = merk.foreign_call @merk_foreign_foreign__proc_Dsucc__2__0(%arg0) : (i64) -> i64
% MERK-NEXT: merk.succeed %[[Y]] : i64
% MERK-LABEL: merk.proc private @foreign_proc.divide_4_0(%arg0: i64, %arg1: i64) -> (i64, i64)
% MERK-NEXT: %[[QR:.*]]:2 = merk.foreign_call @merk_foreign_foreign__proc_Ddivide__4__0(%arg0, %arg1) : (i64, i64) -> (i64, i64)
% MERK-NEXT: merk.succeed %[[QR]]#0, %[[QR]]#1 : i64, i64
% MERK-LABEL: merk.proc private @foreign_proc.length_f1_0(%arg0: !merk.string) -> i64
% MERK-LABEL: merk.proc private @foreign_proc.show_3_0(%arg0: i64)

% LLVM-LABEL: define internal tailcc { i64, i64 } @foreign_proc.divide_4_0(i64 %0, i64 %1)
% LLVM: alloca i64
% LLVM: alloca i64
% LLVM: call void @merk_foreign_foreign__proc_Ddivide__4__0(i64 %0, i64 %1, ptr %{{.*}}, ptr %{{.*}})
% LLVM: define internal i64 @merk_foreign_foreign__proc_Dsucc__2__0(i64 noundef %X)
% LLVM: define internal void @merk_foreign_foreign__proc_Ddivide__4__0(

% OUTPUT: 42 3 1 5

:- module foreign_proc.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

main(!IO) :-
    succ(41, N),
    divide(7, 2, Q, R),
    show(N, !IO),
    show(Q, !IO),
    show(R, !IO),
    show(length("hello"), !IO),
    io.nl(!IO).

:- pragma foreign_decl("C", "#include <stdio.h>").
:- pragma foreign_decl("C", "#include <string.h>").

:- pred succ(int::in, int::out) is det.
:- pragma foreign_proc("C",
    succ(X::in, Y::out),
    [will_not_call_mercury, promise_pure],
"Y = X + 1;").

:- pred divide(int::in, int::in, int::out, int::out) is det.
:- pragma foreign_proc("C",
    divide(A::in, B::in, Q::out, R::out),
    [will_not_call_mercury, promise_pure],
"Q = A / B; R = A % B;").

:- func length(string) = int.
:- pragma foreign_proc("C",
    length(S::in) = (N::out),
    [will_not_call_mercury, promise_pure],
"N = (MR_Integer) strlen(S);").

:- pred show(int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    show(N::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"printf(\"%ld \", (long) N);").
