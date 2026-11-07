% A receive goal's expansion uses the module process, which the module
% must import (design document, "The receive goal").
%
% RUN: not merkc -fc1 -freceive-goal -fdebug-dump-determinism %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module import.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

main(!IO) :-
    receive [io(!IO)] ( select(_, N) -> io.write_int(N, !IO) ).
% CHECK: import.m:[[@LINE-1]]:5: error: a receive goal needs the module process, which this module does not import
