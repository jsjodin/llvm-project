% -fmercury-messages marks the entry, and lowering then defines
% merk_mercury_messages, which replaces the runtime's weak 0: an uncaught
% exception is reported as mmc's programs report it. Without the flag the
% report names Merkurius.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu -fmercury-messages %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu -fmercury-messages %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=NONE
% RUN: merkc -O0 -fmercury-messages %s -o %t1
% RUN: not %t1 2>&1 | FileCheck %s --check-prefix=MERCURY --match-full-lines --strict-whitespace
% RUN: merkc -O0 %s -o %t0
% RUN: not %t0 2>&1 | FileCheck %s --check-prefix=MERKURIUS --match-full-lines --strict-whitespace

:- module mercury_messages.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module require.

main(!IO) :-
    io.write_string("before\n", !IO),
    error("gave up").

% MERK: merk.entry @mercury_messages.main_2_0 mercury_messages
% LLVM: @merk_mercury_messages = constant i32 1
% NONE-NOT: merk_mercury_messages

% MERCURY:before
% MERCURY-NEXT:Uncaught Mercury exception:
% MERCURY-NEXT:Software Error: gave up

% MERKURIUS:before
% MERKURIUS-NEXT:Uncaught Merkurius exception:
% MERKURIUS-NEXT:Software Error: gave up
