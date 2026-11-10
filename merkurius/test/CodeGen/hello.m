% Hello, world: a string constant, and a call of io.write_string/3, which
% the compiled standard library defines with a foreign_proc. The string is
% a constant global of its bytes and a NUL.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK: merk.proc @hello.main_2_0() attributes {determinism = #merk.determinism<det>} {
% MERK-NEXT: %[[S:.*]] = merk.constant "Hello, world!\0A" : !merk.string
% MERK-NEXT: merk.call @io.write_string_3_0(%[[S]]) : (!merk.string) -> ()
% MERK-NEXT: merk.succeed
% MERK: merk.proc private @io.write_string_3_0(!merk.string)
% MERK: merk.entry @hello.main_2_0

% LLVM: @merk.str.0 = private unnamed_addr constant [15 x i8] c"Hello, world!\0A\00"
% LLVM: define tailcc void @hello.main_2_0() {
% LLVM: call tailcc void @io.write_string_3_0(ptr addrspace(1) addrspacecast (ptr @merk.str.0 to ptr addrspace(1)))
% LLVM: declare tailcc void @io.write_string_3_0(ptr addrspace(1))

% OUTPUT:Hello, world!
% OUTPUT-EMPTY:

:- module hello.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
main(!IO) :-
    io.write_string("Hello, world!\n", !IO).
