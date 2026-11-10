% An imported module is read with the extensions of the compile that imports
% it, so that it may use do loops (-fdo-loops) and receive goals
% (-freceive-goal). mmc has neither, so the output is written by hand.
%
% RUN: merkc -fdo-loops -freceive-goal -I %S/Inputs %S/Inputs/loop_sum.m %S/Inputs/first_message.m %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -freceive-goal -I %S/Inputs %S/Inputs/loop_sum.m %S/Inputs/first_message.m %s -o %t.O2 && %t.O2 | FileCheck %s --match-full-lines --strict-whitespace

:- module extension_modules.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module first_message, list, loop_sum, process.

main(!IO) :-
    io.write_int(loop_sum([1, 2, 3]), !IO),
    io.nl(!IO),
    new_mailbox(Mailbox, !IO),
    first_message(Mailbox, None, !IO),
    send(Mailbox, 7, !IO),
    first_message(Mailbox, Seven, !IO),
    io.write_int(None, !IO),
    io.nl(!IO),
    io.write_int(Seven, !IO),
    io.nl(!IO).

% CHECK:6
% CHECK-NEXT:-1
% CHECK-NEXT:7
