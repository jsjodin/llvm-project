% Forms that desugaring does not handle yet are reported and left out. The
% module compiles with mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-semantic-form %s 2>&1 | FileCheck %s

:- module not_yet.
:- interface.
:- import_module io.

:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module int.

main(!IO) :-
    io.write_string("hello\n", !IO).

:- mutable(counter, int, 0, ground, [untrailed, attach_to_io_state]).
% CHECK: not-yet.m:[[@LINE-1]]:4: error: not yet implemented: desugaring mutable declarations
