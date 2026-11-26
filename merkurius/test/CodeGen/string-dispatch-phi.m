% A dispatch on the command line's words, each test comparing the first
% with a string literal: at -O2, GVN's PRE merges the C pointers made from
% the same heap string for the comparisons into a phi, which the collector
% rewrite took for a C pointer it could not make again at its uses. The
% phi becomes one of the heap pointers, cast after it, so that a collection
% in between, as one in the first test's condition, which allocates,
% relocates what the comparison reads.
%
% RUN: merkc -O0 %s -o %t0
% RUN: merkc -O2 %s -o %t2
% RUN: %t0 count 5 | FileCheck %s --check-prefix=COUNT --match-full-lines
% RUN: %t2 count 5 | FileCheck %s --check-prefix=COUNT --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 count 5 | FileCheck %s --check-prefix=COUNT --match-full-lines
% RUN: %t2 name merkurius | FileCheck %s --check-prefix=NAME --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 name merkurius | FileCheck %s --check-prefix=NAME --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 count x | FileCheck %s --check-prefix=USAGE --match-full-lines
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 count x | FileCheck %s --check-prefix=USAGE --match-full-lines
% RUN: %t2 | FileCheck %s --check-prefix=USAGE --match-full-lines
% COUNT: 50
% NAME: merkurius
% USAGE: usage

:- module string_dispatch_phi.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if
        Args = ["count", Text],
        string.to_int(Text ++ "0", N),
        N >= 0
    then
        io.write_int(N, !IO),
        io.nl(!IO)
    else if Args = ["name", Name] then
        io.write_string(Name, !IO),
        io.nl(!IO)
    else
        io.write_string("usage\n", !IO)
    ).
